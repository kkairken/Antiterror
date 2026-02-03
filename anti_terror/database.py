"""Database module for persistent storage of faces, bags, and ownership.

Uses SQLite for simplicity and portability.

Features:
- Identity embeddings for cross-session re-identification
- Track ID history for relink after ByteTrack loss
- Session continuity for 24/7 operation
"""
from __future__ import annotations

import sqlite3
import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from loguru import logger


@dataclass
class PersonRecord:
    """Record of a known person."""
    person_id: str
    first_seen: float
    last_seen: float
    embedding_count: int = 0
    best_image_path: str | None = None


@dataclass
class BagRecord:
    """Record of a known bag."""
    bag_id: str
    owner_person_id: str | None
    first_seen: float
    last_seen: float
    image_path: str | None = None
    is_abandoned: bool = False


@dataclass
class OwnershipRecord:
    """Record of bag-person ownership relationship."""
    bag_id: str
    person_id: str
    confidence: float  # How confident we are in this ownership (0-1)
    first_linked: float
    last_confirmed: float
    link_count: int = 1  # Number of times this link was confirmed


class Database:
    """SQLite database for storing faces, bags, and ownership relationships."""

    def __init__(self, db_path: str = "antiterror.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Database initialized at {self.db_path}")

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Persons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                embedding_count INTEGER DEFAULT 0,
                best_image_path TEXT
            )
        """)

        # Bags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bags (
                bag_id TEXT PRIMARY KEY,
                owner_person_id TEXT,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                image_path TEXT,
                is_abandoned INTEGER DEFAULT 0,
                FOREIGN KEY (owner_person_id) REFERENCES persons(person_id)
            )
        """)

        # Ownership history table (tracks all ownership links)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ownership (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bag_id TEXT NOT NULL,
                person_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                first_linked REAL NOT NULL,
                last_confirmed REAL NOT NULL,
                link_count INTEGER DEFAULT 1,
                FOREIGN KEY (bag_id) REFERENCES bags(bag_id),
                FOREIGN KEY (person_id) REFERENCES persons(person_id),
                UNIQUE(bag_id, person_id)
            )
        """)

        # Events table (for logging)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                bag_id TEXT,
                person_id TEXT,
                details TEXT
            )
        """)

        # Embeddings table (optional - for re-identification across sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,  -- 'person' or 'bag'
                entity_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                quality REAL DEFAULT 1.0,
                timestamp REAL NOT NULL
            )
        """)

        # Identity embeddings table (for gallery persistence)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS identity_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                quality REAL DEFAULT 1.0,
                timestamp REAL NOT NULL,
                is_centroid INTEGER DEFAULT 0,
                UNIQUE(person_id, timestamp)
            )
        """)

        # Track ID history (for relink after occlusion)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_id_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                session_id TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        """)

        # Sessions table (for session continuity)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                camera_id TEXT,
                metadata TEXT
            )
        """)

        # Bag ownership snapshots (for persistence across restarts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bag_ownership_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bag_id TEXT NOT NULL,
                bag_track_id INTEGER,
                owner_person_id TEXT,
                confidence REAL DEFAULT 0.0,
                is_validated INTEGER DEFAULT 0,
                first_seen REAL,
                last_seen REAL,
                session_id TEXT,
                snapshot_time REAL NOT NULL,
                FOREIGN KEY (owner_person_id) REFERENCES persons(person_id)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_identity_embeddings_person ON identity_embeddings(person_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_history_person ON track_id_history(person_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bag_ownership_session ON bag_ownership_snapshots(session_id)")

        self.conn.commit()

    # === Person operations ===

    def add_person(self, person_id: str, image_path: str | None = None) -> PersonRecord:
        """Add or update a person record."""
        now = time.time()
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO persons (person_id, first_seen, last_seen, embedding_count, best_image_path)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                last_seen = ?,
                embedding_count = embedding_count + 1
        """, (person_id, now, now, image_path, now))

        self.conn.commit()
        return self.get_person(person_id)

    def get_person(self, person_id: str) -> PersonRecord | None:
        """Get person record by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
        row = cursor.fetchone()
        if row:
            return PersonRecord(
                person_id=row['person_id'],
                first_seen=row['first_seen'],
                last_seen=row['last_seen'],
                embedding_count=row['embedding_count'],
                best_image_path=row['best_image_path']
            )
        return None

    def get_all_persons(self) -> List[PersonRecord]:
        """Get all person records."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM persons ORDER BY last_seen DESC")
        return [
            PersonRecord(
                person_id=row['person_id'],
                first_seen=row['first_seen'],
                last_seen=row['last_seen'],
                embedding_count=row['embedding_count'],
                best_image_path=row['best_image_path']
            )
            for row in cursor.fetchall()
        ]

    # === Bag operations ===

    def add_bag(self, bag_id: str, owner_person_id: str | None = None,
                image_path: str | None = None) -> BagRecord:
        """Add or update a bag record."""
        now = time.time()
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO bags (bag_id, owner_person_id, first_seen, last_seen, image_path)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(bag_id) DO UPDATE SET
                last_seen = ?,
                owner_person_id = COALESCE(?, owner_person_id)
        """, (bag_id, owner_person_id, now, now, image_path, now, owner_person_id))

        self.conn.commit()
        return self.get_bag(bag_id)

    def get_bag(self, bag_id: str) -> BagRecord | None:
        """Get bag record by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bags WHERE bag_id = ?", (bag_id,))
        row = cursor.fetchone()
        if row:
            return BagRecord(
                bag_id=row['bag_id'],
                owner_person_id=row['owner_person_id'],
                first_seen=row['first_seen'],
                last_seen=row['last_seen'],
                image_path=row['image_path'],
                is_abandoned=bool(row['is_abandoned'])
            )
        return None

    def get_bags_by_owner(self, person_id: str) -> List[BagRecord]:
        """Get all bags owned by a person."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bags WHERE owner_person_id = ?", (person_id,))
        return [
            BagRecord(
                bag_id=row['bag_id'],
                owner_person_id=row['owner_person_id'],
                first_seen=row['first_seen'],
                last_seen=row['last_seen'],
                image_path=row['image_path'],
                is_abandoned=bool(row['is_abandoned'])
            )
            for row in cursor.fetchall()
        ]

    def set_bag_abandoned(self, bag_id: str, is_abandoned: bool = True):
        """Mark a bag as abandoned or not."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE bags SET is_abandoned = ? WHERE bag_id = ?",
            (int(is_abandoned), bag_id)
        )
        self.conn.commit()

    # === Ownership operations ===

    def link_bag_to_person(self, bag_id: str, person_id: str, confidence: float = 1.0):
        """Create or update ownership link between bag and person."""
        now = time.time()
        cursor = self.conn.cursor()

        # Update ownership table
        cursor.execute("""
            INSERT INTO ownership (bag_id, person_id, confidence, first_linked, last_confirmed, link_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(bag_id, person_id) DO UPDATE SET
                confidence = MAX(confidence, ?),
                last_confirmed = ?,
                link_count = link_count + 1
        """, (bag_id, person_id, confidence, now, now, confidence, now))

        # Update bag's primary owner (highest confidence)
        cursor.execute("""
            UPDATE bags SET owner_person_id = (
                SELECT person_id FROM ownership
                WHERE bag_id = ?
                ORDER BY confidence DESC, link_count DESC
                LIMIT 1
            ) WHERE bag_id = ?
        """, (bag_id, bag_id))

        self.conn.commit()

    def get_bag_owner(self, bag_id: str) -> str | None:
        """Get the primary owner of a bag."""
        bag = self.get_bag(bag_id)
        return bag.owner_person_id if bag else None

    def get_ownership_history(self, bag_id: str) -> List[OwnershipRecord]:
        """Get all ownership records for a bag."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM ownership
            WHERE bag_id = ?
            ORDER BY confidence DESC, link_count DESC
        """, (bag_id,))

        return [
            OwnershipRecord(
                bag_id=row['bag_id'],
                person_id=row['person_id'],
                confidence=row['confidence'],
                first_linked=row['first_linked'],
                last_confirmed=row['last_confirmed'],
                link_count=row['link_count']
            )
            for row in cursor.fetchall()
        ]

    # === Event logging ===

    def log_event(self, event_type: str, bag_id: str | None = None,
                  person_id: str | None = None, details: dict | None = None):
        """Log an event to the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO events (timestamp, event_type, bag_id, person_id, details)
            VALUES (?, ?, ?, ?, ?)
        """, (time.time(), event_type, bag_id, person_id,
              json.dumps(details) if details else None))
        self.conn.commit()

    def get_recent_events(self, limit: int = 100) -> List[dict]:
        """Get recent events."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM events
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        events = []
        for row in cursor.fetchall():
            events.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'event_type': row['event_type'],
                'bag_id': row['bag_id'],
                'person_id': row['person_id'],
                'details': json.loads(row['details']) if row['details'] else None
            })
        return events

    # === Embedding storage (for cross-session re-identification) ===

    def store_embedding(self, entity_type: str, entity_id: str,
                        embedding: torch.Tensor, quality: float = 1.0):
        """Store an embedding for later re-identification."""
        cursor = self.conn.cursor()
        embedding_bytes = embedding.cpu().numpy().tobytes()
        cursor.execute("""
            INSERT INTO embeddings (entity_type, entity_id, embedding, quality, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (entity_type, entity_id, embedding_bytes, quality, time.time()))
        self.conn.commit()

    def get_embeddings(self, entity_type: str, entity_id: str) -> List[torch.Tensor]:
        """Get stored embeddings for an entity."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT embedding FROM embeddings
            WHERE entity_type = ? AND entity_id = ?
            ORDER BY quality DESC
            LIMIT 10
        """, (entity_type, entity_id))

        embeddings = []
        for row in cursor.fetchall():
            arr = np.frombuffer(row['embedding'], dtype=np.float32)
            embeddings.append(torch.from_numpy(arr.copy()))
        return embeddings

    # === Identity Gallery Persistence ===

    def save_identity(
        self,
        person_id: str,
        embeddings: List[torch.Tensor],
        qualities: List[float],
        centroid: torch.Tensor | None = None
    ) -> None:
        """Save identity embeddings to database."""
        cursor = self.conn.cursor()
        now = time.time()

        # Save individual embeddings (keep top 10 by quality)
        paired = list(zip(embeddings, qualities))
        paired.sort(key=lambda x: x[1], reverse=True)

        for emb, quality in paired[:10]:
            embedding_bytes = emb.cpu().numpy().tobytes()
            cursor.execute("""
                INSERT OR REPLACE INTO identity_embeddings
                (person_id, embedding, quality, timestamp, is_centroid)
                VALUES (?, ?, ?, ?, 0)
            """, (person_id, embedding_bytes, quality, now))

        # Save centroid separately if provided
        if centroid is not None:
            centroid_bytes = centroid.cpu().numpy().tobytes()
            cursor.execute("""
                INSERT OR REPLACE INTO identity_embeddings
                (person_id, embedding, quality, timestamp, is_centroid)
                VALUES (?, ?, 1.0, ?, 1)
            """, (person_id, centroid_bytes, now))

        self.conn.commit()

    def load_gallery(self, max_identities: int = 500, ttl_hours: float = 8.0) -> Dict[str, dict]:
        """Load identity gallery from database.

        Returns dict of person_id -> {embeddings, qualities, centroid, last_seen}
        """
        cursor = self.conn.cursor()
        now = time.time()
        cutoff = now - (ttl_hours * 3600)

        # Get recent persons
        cursor.execute("""
            SELECT person_id, last_seen FROM persons
            WHERE last_seen > ?
            ORDER BY last_seen DESC
            LIMIT ?
        """, (cutoff, max_identities))

        person_rows = cursor.fetchall()
        gallery: Dict[str, dict] = {}

        for row in person_rows:
            person_id = row['person_id']
            last_seen = row['last_seen']

            # Get embeddings for this person
            cursor.execute("""
                SELECT embedding, quality, is_centroid FROM identity_embeddings
                WHERE person_id = ?
                ORDER BY is_centroid DESC, quality DESC
                LIMIT 30
            """, (person_id,))

            embeddings = []
            qualities = []
            centroid = None

            for emb_row in cursor.fetchall():
                arr = np.frombuffer(emb_row['embedding'], dtype=np.float32)
                tensor = torch.from_numpy(arr.copy())

                if emb_row['is_centroid']:
                    centroid = tensor
                else:
                    embeddings.append(tensor)
                    qualities.append(emb_row['quality'])

            if embeddings:
                gallery[person_id] = {
                    'embeddings': embeddings,
                    'qualities': qualities,
                    'centroid': centroid,
                    'last_seen': last_seen
                }

        logger.info(f"Loaded {len(gallery)} identities from database")
        return gallery

    def save_gallery_batch(self, identities: Dict[str, 'FaceIdentity']) -> None:
        """Save multiple identities in a batch."""
        for person_id, identity in identities.items():
            self.save_identity(
                person_id=person_id,
                embeddings=identity.embeddings,
                qualities=identity.qualities,
                centroid=identity.centroid
            )

    # === Track ID History ===

    def save_track_mapping(
        self,
        person_id: str,
        track_id: int,
        start_time: float,
        end_time: float | None = None,
        session_id: str | None = None
    ) -> None:
        """Save track_id to person_id mapping."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO track_id_history
            (person_id, track_id, start_time, end_time, session_id)
            VALUES (?, ?, ?, ?, ?)
        """, (person_id, track_id, start_time, end_time, session_id))
        self.conn.commit()

    def get_recent_track_history(
        self,
        window_hours: float = 1.0
    ) -> List[Tuple[str, int, float, float | None]]:
        """Get recent track history for relink."""
        cursor = self.conn.cursor()
        cutoff = time.time() - (window_hours * 3600)

        cursor.execute("""
            SELECT person_id, track_id, start_time, end_time
            FROM track_id_history
            WHERE start_time > ?
            ORDER BY start_time DESC
        """, (cutoff,))

        return [(row['person_id'], row['track_id'], row['start_time'], row['end_time'])
                for row in cursor.fetchall()]

    # === Session Management ===

    def start_session(self, camera_id: str | None = None) -> str:
        """Start a new session and return session_id."""
        session_id = str(uuid.uuid4())[:8]
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, start_time, camera_id)
            VALUES (?, ?, ?)
        """, (session_id, time.time(), camera_id))
        self.conn.commit()
        logger.info(f"Started session {session_id}")
        return session_id

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions SET end_time = ? WHERE session_id = ?
        """, (time.time(), session_id))
        self.conn.commit()

    def get_last_session_id(self) -> str | None:
        """Get the most recent session ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1
        """)
        row = cursor.fetchone()
        return row['session_id'] if row else None

    # === Bag Ownership Persistence ===

    def save_bag_ownership(
        self,
        bag_id: str,
        bag_track_id: int,
        owner_person_id: str | None,
        confidence: float,
        is_validated: bool,
        first_seen: float,
        last_seen: float,
        session_id: str | None = None
    ) -> None:
        """Save bag ownership snapshot."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO bag_ownership_snapshots
            (bag_id, bag_track_id, owner_person_id, confidence, is_validated,
             first_seen, last_seen, session_id, snapshot_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (bag_id, bag_track_id, owner_person_id, confidence,
              int(is_validated), first_seen, last_seen, session_id, time.time()))
        self.conn.commit()

    def load_bag_ownerships(
        self,
        session_id: str | None = None,
        max_age_hours: float = 8.0
    ) -> List[dict]:
        """Load bag ownership snapshots.

        Returns list of ownership dicts with bag_id, owner_person_id, confidence, etc.
        """
        cursor = self.conn.cursor()
        cutoff = time.time() - (max_age_hours * 3600)

        if session_id:
            cursor.execute("""
                SELECT * FROM bag_ownership_snapshots
                WHERE session_id = ? AND snapshot_time > ?
                ORDER BY snapshot_time DESC
            """, (session_id, cutoff))
        else:
            cursor.execute("""
                SELECT * FROM bag_ownership_snapshots
                WHERE snapshot_time > ?
                ORDER BY snapshot_time DESC
            """, (cutoff,))

        ownerships = []
        seen_bags = set()

        for row in cursor.fetchall():
            bag_id = row['bag_id']
            if bag_id in seen_bags:
                continue  # Skip older snapshots for same bag
            seen_bags.add(bag_id)

            ownerships.append({
                'bag_id': bag_id,
                'bag_track_id': row['bag_track_id'],
                'owner_person_id': row['owner_person_id'],
                'confidence': row['confidence'],
                'is_validated': bool(row['is_validated']),
                'first_seen': row['first_seen'],
                'last_seen': row['last_seen']
            })

        logger.info(f"Loaded {len(ownerships)} bag ownerships from database")
        return ownerships

    def save_ownerships_batch(
        self,
        ownerships: Dict[int, 'BagOwnership'],
        session_id: str | None = None
    ) -> None:
        """Save multiple bag ownerships in a batch."""
        for bag_track_id, ownership in ownerships.items():
            if ownership.bag_id and ownership.owner_person_id:
                self.save_bag_ownership(
                    bag_id=ownership.bag_id,
                    bag_track_id=bag_track_id,
                    owner_person_id=ownership.owner_person_id,
                    confidence=ownership.confidence,
                    is_validated=ownership.validated,
                    first_seen=ownership.first_seen,
                    last_seen=ownership.last_owner_seen,
                    session_id=session_id
                )

    # === Cleanup ===

    def cleanup_old_data(self, max_age_hours: float = 24.0) -> dict:
        """Clean up old data to prevent unbounded growth.

        Returns dict with counts of deleted rows.
        """
        cursor = self.conn.cursor()
        cutoff = time.time() - (max_age_hours * 3600)
        deleted = {}

        # Clean old identity embeddings
        cursor.execute("DELETE FROM identity_embeddings WHERE timestamp < ?", (cutoff,))
        deleted['identity_embeddings'] = cursor.rowcount

        # Clean old track history
        cursor.execute("DELETE FROM track_id_history WHERE start_time < ?", (cutoff,))
        deleted['track_id_history'] = cursor.rowcount

        # Clean old ownership snapshots
        cursor.execute("DELETE FROM bag_ownership_snapshots WHERE snapshot_time < ?", (cutoff,))
        deleted['bag_ownership_snapshots'] = cursor.rowcount

        # Clean old events
        cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        deleted['events'] = cursor.rowcount

        self.conn.commit()

        if sum(deleted.values()) > 0:
            logger.info(f"Cleaned up old data: {deleted}")

        return deleted

    # === Statistics ===

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM persons")
        person_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM bags")
        bag_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM bags WHERE is_abandoned = 1")
        abandoned_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ownership")
        link_count = cursor.fetchone()[0]

        return {
            'persons': person_count,
            'bags': bag_count,
            'abandoned_bags': abandoned_count,
            'ownership_links': link_count
        }

    def close(self):
        """Close database connection."""
        self.conn.close()

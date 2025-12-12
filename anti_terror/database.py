"""Database module for persistent storage of faces, bags, and ownership.

Uses SQLite for simplicity and portability.
"""
from __future__ import annotations

import sqlite3
import json
import time
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

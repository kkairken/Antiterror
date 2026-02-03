import os
import sys

# Fix SSL certificate verification on macOS
# This sets the certificate bundle path for urllib/requests
if sys.platform == "darwin":
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    except ImportError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

from anti_terror.runner import main


if __name__ == "__main__":
    main()

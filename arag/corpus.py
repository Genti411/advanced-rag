"""A small security knowledge base + a labeled retrieval eval set.

Each eval query has one known-relevant doc id. Some queries use exact keywords
(favor BM25), others paraphrase (favor dense) — so hybrid retrieval should beat
either method alone.
"""

DOCS = {
    "least_privilege": "The principle of least privilege grants each user, process, and service only the minimum access and permissions required to do its job, bounding the damage from a compromised account.",
    "incident_response": "The SANS PICERL incident-response lifecycle has six phases: Preparation, Identification, Containment, Eradication, Recovery, and Lessons Learned.",
    "mitre_attack": "MITRE ATT&CK is a knowledge base of adversary tactics and techniques; technique T1110 is Brute Force and T1059 is Command and Scripting Interpreter.",
    "password_storage": "Store passwords with a slow salted hashing function such as bcrypt, scrypt, or Argon2; a unique salt per password defeats rainbow-table attacks.",
    "zero_trust": "Zero Trust security assumes no implicit trust based on network location and continuously verifies every request: never trust, always verify.",
    "mfa": "Multi-factor authentication requires a second factor in addition to a password, so a stolen password alone is not enough to gain access.",
    "encryption_tls": "TLS encrypts data in transit; during the handshake the server presents a certificate and the parties negotiate keys before exchanging application data.",
    "xss": "Cross-site scripting (XSS) injects attacker-controlled JavaScript into a web page; contextual output encoding and a Content-Security-Policy mitigate it.",
    "sql_injection": "SQL injection occurs when untrusted input is concatenated into a query; parameterized queries and prepared statements prevent it.",
    "ddos": "A distributed denial-of-service attack floods a target with traffic from many sources to exhaust its resources and make it unavailable.",
    "firewall": "A firewall filters network traffic by rules; a next-generation firewall adds application awareness and intrusion prevention.",
    "backup_recovery": "Regular tested backups following the 3-2-1 rule enable recovery from ransomware and data loss; RPO and RTO define acceptable data loss and downtime.",
}

EVAL = [
    ("what does minimal access and permissions mean", "least_privilege"),
    ("PICERL incident response phases preparation containment eradication", "incident_response"),
    ("T1110 brute force adversary technique", "mitre_attack"),
    ("how should I store passwords securely with bcrypt and a salt", "password_storage"),
    ("never trust always verify every request", "zero_trust"),
    ("second factor beyond a password to log in", "mfa"),
    ("TLS handshake certificate encrypt data in transit", "encryption_tls"),
    ("attacker injected javascript runs in the browser", "xss"),
    ("parameterized queries prevent injecting SQL", "sql_injection"),
    ("flood a server with traffic from many machines", "ddos"),
    # --- adversarial: synonym paraphrases (keyword-sparse; hard for BM25) ---
    ("limit each worker to the smallest set of rights they need", "least_privilege"),
    ("confirm who someone is with something beyond a secret phrase", "mfa"),
    ("keep a tamper-proof trail of security events for later review", "audit_logging"),
    ("scramble information so eavesdroppers cannot read it", "encryption"),
    # --- adversarial: bare identifier (semantically thin; hard for dense) ---
    ("T1059 command interpreter technique", "mitre_attack"),
    ("overwhelm a site with junk traffic from many machines", "ddos"),
]

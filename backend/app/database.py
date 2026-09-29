"""
BlockNexa — MongoDB Database Integration
========================================
Supports local MongoDB (mongodb://localhost:27017) and cloud MongoDB Atlas
with resilient fallback if database is not reachable.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pymongo
from pymongo import MongoClient

# Environment configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "blocknexa")


class DatabaseManager:
    def __init__(self):
        self.uri = MONGODB_URI
        self.db_name = MONGODB_DB_NAME
        self.client: Optional[MongoClient] = None
        self.db = None
        self.is_connected = False
        self._init_connection()

    def _init_connection(self):
        try:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=2500,
                connectTimeoutMS=2500
            )
            # Force server roundtrip to verify connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.is_connected = True
            print(f"[OK] MongoDB Connected successfully to database '{self.db_name}' at {self.uri.split('@')[-1] if '@' in self.uri else self.uri}")
        except Exception as e:
            self.is_connected = False
            self.db = None
            print(f"[!] MongoDB not reachable ({e}). Running in fallback mode.")

    def get_status(self) -> Dict[str, Any]:
        """Returns database connectivity and collection stats."""
        if not self.is_connected or self.db is None:
            # Recheck connection
            self._init_connection()

        if self.is_connected and self.db is not None:
            try:
                collections = self.db.list_collection_names()
                return {
                    "connected": True,
                    "database": self.db_name,
                    "uri": self.uri.split('@')[-1] if '@' in self.uri else self.uri,
                    "collections": collections,
                    "counts": {c: self.db[c].count_documents({}) for c in collections}
                }
            except Exception as e:
                return {"connected": False, "error": str(e)}

        return {
            "connected": False,
            "database": self.db_name,
            "uri": self.uri.split('@')[-1] if '@' in self.uri else self.uri,
            "note": "MongoDB is not reachable or offline. In-memory fallback active."
        }

    def save_block_plan(self, plan_data: Dict[str, Any]) -> Optional[str]:
        """Saves generated optimization block plan to MongoDB."""
        if not self.is_connected or self.db is None:
            return None
        try:
            doc = {
                **plan_data,
                "created_at": datetime.utcnow().isoformat()
            }
            res = self.db.block_plans.insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            print(f"[!] Error saving block plan to MongoDB: {e}")
            return None

    def get_recent_block_plans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves recent saved block plans."""
        if not self.is_connected or self.db is None:
            return []
        try:
            cursor = self.db.block_plans.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"[!] Error fetching block plans: {e}")
            return []

    def save_defect_prediction(self, prediction_data: Dict[str, Any]) -> Optional[str]:
        """Logs ML defect risk predictions."""
        if not self.is_connected or self.db is None:
            return None
        try:
            doc = {
                **prediction_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            res = self.db.defect_predictions.insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            print(f"[!] Error saving defect prediction: {e}")
            return None

    def save_delay_prediction(self, delay_data: Dict[str, Any]) -> Optional[str]:
        """Logs ML train delay predictions."""
        if not self.is_connected or self.db is None:
            return None
        try:
            doc = {
                **delay_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            res = self.db.delay_predictions.insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            print(f"[!] Error saving delay prediction: {e}")
            return None

    def seed_initial_data(self) -> Dict[str, Any]:
        """Seeds initial collections with sample railway reference & history data."""
        if not self.is_connected or self.db is None:
            return {"status": "error", "message": "Database not connected"}
        
        try:
            import pandas as pd
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

            # Seed corridor reference if empty
            if self.db.corridors.count_documents({}) == 0:
                self.db.corridors.insert_many([
                    {"section": "BSL-JL", "line": "UP/DN", "length_km": 24, "speed_limit": 130, "traction": "25kV AC"},
                    {"section": "MMR-CSN", "line": "UP/DN", "length_km": 42, "speed_limit": 110, "traction": "25kV AC"},
                    {"section": "CSN-BSL", "line": "UP/DN", "length_km": 80, "speed_limit": 120, "traction": "25kV AC"},
                    {"section": "JL-NGN", "line": "UP/DN", "length_km": 35, "speed_limit": 110, "traction": "25kV AC"}
                ])

            # Seed maintenance tasks if empty
            tasks_csv = os.path.join(data_dir, "maintenance_tasks.csv")
            if os.path.exists(tasks_csv) and self.db.maintenance_tasks.count_documents({}) == 0:
                df = pd.read_csv(tasks_csv)
                self.db.maintenance_tasks.insert_many(df.to_dict("records"))

            # Seed block history if empty
            history_csv = os.path.join(data_dir, "block_history.csv")
            if os.path.exists(history_csv) and self.db.block_history.count_documents({}) == 0:
                df = pd.read_csv(history_csv)
                self.db.block_history.insert_many(df.to_dict("records"))

            # Seed system audit log
            self.db.audit_logs.insert_one({
                "action": "DATABASE_INITIALIZED",
                "timestamp": datetime.utcnow().isoformat(),
                "system": "BlockNexa AI Automatic Railway Block Planning Engine",
                "division": "Central Railway • Bhusawal Division [BSL]"
            })

            return {
                "status": "success",
                "message": "BlockNexa MongoDB collections initialized and seeded successfully",
                "collections": self.db.list_collection_names()
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


db_manager = DatabaseManager()

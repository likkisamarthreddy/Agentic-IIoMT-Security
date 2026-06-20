import logging
import uuid
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("iimt.system2.sdn_controller")

class SDNController:
    """SDN action executor for simulating OpenFlow-style mitigation rules."""
    
    def __init__(self):
        self.flow_table: Dict[str, Dict] = {}
        logger.info("SDN Controller initialized.")

    def apply_rule(self, mitigation_action: dict) -> str:
        """Applies an SDN flow rule and returns the rule ID."""
        rule_id = f"SDN-{uuid.uuid4().hex[:8]}"
        device_id = mitigation_action.get("device_id")
        action_name = mitigation_action.get("action_name", "LOG_ONLY")
        
        rule = {
            "rule_id": rule_id,
            "device_id": device_id,
            "action_type": action_name,
            "parameters": mitigation_action.get("parameters", {}),
            "timestamp": datetime.now().isoformat(),
            "active": True
        }
        
        self.flow_table[rule_id] = rule
        logger.info(f"Applied SDN Rule {rule_id}: {action_name} for device {device_id}")
        return rule_id

    def rollback_rule(self, device_id: str, rule_id: str) -> bool:
        """Removes/reverses a specific rule."""
        if rule_id in self.flow_table and self.flow_table[rule_id]["device_id"] == device_id:
            self.flow_table[rule_id]["active"] = False
            logger.info(f"Rolled back SDN Rule {rule_id} for device {device_id}")
            return True
        return False

    def rollback_all_rules(self, device_id: str) -> int:
        """Removes/reverses all rules for a specific device."""
        count = 0
        for rule_id, rule in self.flow_table.items():
            if rule["device_id"] == device_id and rule["active"]:
                rule["active"] = False
                count += 1
        if count > 0:
            logger.info(f"Rolled back {count} SDN rules for device {device_id}")
        return count

    def get_active_rules(self, device_id: str) -> List[Dict]:
        """Returns active rules for a specific device."""
        return [rule for rule in self.flow_table.values() if rule["device_id"] == device_id and rule["active"]]

    def get_all_rules(self) -> List[Dict]:
        """Returns all active flow rules."""
        return [rule for rule in self.flow_table.values() if rule["active"]]

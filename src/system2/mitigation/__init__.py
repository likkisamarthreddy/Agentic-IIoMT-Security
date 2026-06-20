# -*- coding: utf-8 -*-
"""
system2.mitigation — Mitigation Sub-package
=============================================

Contains graduated mitigation action components:

* **ActionPlaybook** — maps risk scores to graduated mitigation levels
  defined in ``safety_policies.yaml`` and respects device constraints.
* **SDNController** — simulated SDN (Software-Defined Networking)
  action executor that maintains an in-memory flow-rule table for
  throttle, VLAN isolation, ACL, and drop actions.
"""

from system2.mitigation.action_playbook import ActionPlaybook
from system2.mitigation.sdn_controller import SDNController

__all__ = [
    "ActionPlaybook",
    "SDNController",
]

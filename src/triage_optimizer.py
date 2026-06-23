from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

class MultiEventTriageOptimizer:
    """
    When multiple events are active simultaneously,
    optimally distribute limited officers across them.
    """
    
    def optimize(self, active_events: list, zone_capacity: dict) -> dict:
        """
        active_events: list of dicts with 'event_id', 'stis', 'zone', 'min_officers_needed'
        zone_capacity: dict mapping zone_name -> available_officers
        """
        if not active_events:
            return {'allocation': {}, 'total_deployed': 0, 'triage_active': False, 'narrative': "No active events."}

        n = len(active_events)
        zones = list(zone_capacity.keys())

        # Objective: minimize total weighted under-coverage
        # c_i = STIS_i
        c = [-max(e['stis'], 0.1) for e in active_events] 

        # Zone capacity constraints
        A_zone = []
        b_zone = []
        for zone in zones:
            row = [1 if active_events[i].get('zone', 'unknown') == zone else 0 for i in range(n)]
            # Only add constraint if zone has events
            if sum(row) > 0:
                A_zone.append(row)
                b_zone.append(zone_capacity.get(zone, 0))

        if A_zone:
            constraints = LinearConstraint(A_zone, ub=b_zone)
        else:
            # If zones don't match, just bound by some large number to allow solving
            constraints = LinearConstraint([ [1]*n ], ub=[999])

        bounds = Bounds(
            lb=[e.get('min_officers_needed', 1) for e in active_events],
            ub=[10] * n
        )

        try:
            result = milp(c, constraints=constraints, bounds=bounds, integrality=[1]*n)

            if result.success:
                allocation = {}
                for i, event in enumerate(active_events):
                    assigned = int(round(result.x[i]))
                    min_n = event.get('min_officers_needed', 1)
                    allocation[event['event_id']] = {
                        'officers_assigned': assigned,
                        'min_needed': min_n,
                        'stis': event['stis'],
                        'coverage_ratio': round(assigned / max(min_n, 1), 2),
                        'status': 'FULLY COVERED' if assigned >= min_n else 'PARTIAL - TRIAGE ACTIVE'
                    }
                
                triage_active = any(a['coverage_ratio'] < 1.0 for a in allocation.values())
                
                return {
                    'allocation': allocation,
                    'total_deployed': int(sum(result.x)),
                    'triage_active': triage_active,
                    'narrative': self._generate_narrative(allocation, triage_active)
                }
        except Exception as e:
            # Fallback
            pass
            
        return self._greedy_fallback(active_events, zone_capacity)

    def _greedy_fallback(self, active_events, zone_capacity):
        # Sort by STIS descending
        sorted_evs = sorted(active_events, key=lambda x: x['stis'], reverse=True)
        allocation = {}
        for ev in sorted_evs:
            z = ev.get('zone', 'unknown')
            avail = zone_capacity.get(z, 0)
            needed = ev.get('min_officers_needed', 1)
            assign = min(avail, 10)
            assign = max(assign, needed) if avail >= needed else avail
            
            zone_capacity[z] = max(0, avail - assign)
            
            allocation[ev['event_id']] = {
                'officers_assigned': assign,
                'min_needed': needed,
                'stis': ev['stis'],
                'coverage_ratio': round(assign / max(needed, 1), 2),
                'status': 'FULLY COVERED' if assign >= needed else 'PARTIAL - TRIAGE ACTIVE'
            }
            
        return {
            'allocation': allocation,
            'total_deployed': sum(a['officers_assigned'] for a in allocation.values()),
            'triage_active': any(a['coverage_ratio'] < 1.0 for a in allocation.values()),
            'narrative': "Fallback greedy allocation used."
        }

    def _generate_narrative(self, allocation, triage_active):
        if not triage_active:
            return "Sufficient officers available. All events fully covered."
        else:
            return "TRIAGE ACTIVE: Insufficient officers. Resources diverted to high-STIS events."

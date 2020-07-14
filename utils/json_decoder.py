import json
from typing import Any

from .schedules import PiecewiseSchedule


class MBPOEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, PiecewiseSchedule):
            return {
                'schedule_name': 'PiecewiseSchedule',
                'endpoints': o._endpoints
            }

        return json.JSONEncoder.default(self, o)


class MBPODecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if 'schedule_name' in obj:
            return PiecewiseSchedule(endpoints=obj['endpoints'])
        return obj

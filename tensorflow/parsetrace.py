import glob
import json
import sys

def parse_tracefile(filename):
    
    with open(filename, 'r') as f:
        raw = json.load(f)

    if 'traceEvents' not in raw:
        return None

    traceEvents = raw['traceEvents']
    
    timestamps = (
        (
            event['ts'],
            event['ts'] + event['dur']
        )
        for event in traceEvents
        if 'ts' in event and 'dur' in event
    )
    timestamps = sorted(timestamps, key=lambda x: x[1])

    min_ts = timestamps[0]
    max_ts = timestamps[-1]
    return max_ts[1] - min_ts[0]

durations = []

for filename in glob.glob(sys.argv[1]):
    duration = parse_tracefile(filename)
    durations.append(duration)
    print float(duration) / 1000
    
print 'average:', float(sum(durations)) / len(durations)
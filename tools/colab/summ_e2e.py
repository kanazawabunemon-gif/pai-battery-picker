import sys, json, glob
from collections import Counter
rows = [json.loads(l) for p in sys.argv[1:] for l in open(p) if l.strip()]
H = sum(r['hazard_total'] for r in rows); Rm = sum(r['hazard_removed'] for r in rows)
print('trays', len(rows), 'hazards removed', Rm, '/', H, '= %.0f%%' % (100 * Rm / H),
      '| all removed in %d/%d trays' % (sum(r['hazard_removed'] == r['hazard_total'] for r in rows), len(rows)),
      '| wrong box', sum(r['trash_in_box'] for r in rows), '| actions/tray %.1f' % (sum(r['actions'] for r in rows) / len(rows)),
      '| sim s/tray %.0f' % (sum(r['sim_seconds'] for r in rows) / len(rows)), '| wall s/tray %.0f' % (sum(r['wall_seconds'] for r in rows) / len(rows)))
print('end states', Counter(r['end'] for r in rows))
print('by kind', Counter((k, s) for r in rows for k, s in r['hazards']))

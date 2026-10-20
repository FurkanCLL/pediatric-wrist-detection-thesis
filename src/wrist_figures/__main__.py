import argparse

from .build import build, update_ap_charts

# Targeted chart updates leave existing X-ray examples and other charts untouched.
parser = argparse.ArgumentParser(description='Render figures from saved project evidence')
parser.add_argument('--ap-charts-only', action='store_true', help='Update class-wise A-C AP50 and overall A-F mAP50 only')
args = parser.parse_args()
if args.ap_charts_only:
    update_ap_charts()
else:
    build()

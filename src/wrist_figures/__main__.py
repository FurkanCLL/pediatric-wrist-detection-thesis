import argparse

from .build import build, update_ap_charts

# Targeted chart updates leave existing X-ray examples and other charts untouched.
parser = argparse.ArgumentParser(description='Render figures from saved project evidence')
targets = parser.add_mutually_exclusive_group()
targets.add_argument('--ap-charts-only', action='store_true', help='Update class-wise Experiment A AP50 and overall A-F mAP50 only')
targets.add_argument('--classwise-only', action='store_true', help='Update only class-wise Experiment A AP50')
args = parser.parse_args()
if args.ap_charts_only or args.classwise_only:
    update_ap_charts(classwise_only=args.classwise_only)
else:
    build()

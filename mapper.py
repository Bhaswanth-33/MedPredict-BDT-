#!/usr/bin/env python3
import sys
import csv

reader = csv.reader(sys.stdin)
header = next(reader, None)

for row in reader:
    try:
        hospital_id = row[0]
        region = row[-6]
        area_type = row[-5]
        emergency_risk = row[-4]
        aid_eligible = row[-2]

        key = f"{region}_{area_type}_{emergency_risk}_{aid_eligible}"
        print(f"{key}\t1")
    except IndexError:
        continue

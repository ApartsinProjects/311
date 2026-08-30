# External data

MC311 is derived from public municipal 311 open data. The raw per-city dumps are NOT rehosted here;
they are large and are regenerable from the released collection scripts against each city's portal.

Regenerate the raw corpus with `python collect_311.py --per-city 50000`. Sources:

| City | Source | Dataset |
|---|---|---|
| Baton Rouge | data.brla.gov (Socrata) | 7ixm-mnvx |
| Bloomington IN | data.bloomington.in.gov (Socrata) | aw6y-t4ix |
| Richmond VA | data.richmondgov.com (Socrata) | vgg4-hjn8 |
| Auburn WA | data.auburnwa.gov (Socrata) | bduj-5afh |
| Gainesville FL | data.cityofgainesville.org (Socrata) | 78uv-94ar |
| Honolulu | data.honolulu.gov (Socrata) | jdy7-ftwe |
| San Francisco | Open311 API (mobile311.sfgov.org) | GeoReport v2 |

Each source is under its city's open-data license (for example San Francisco Open311 under PDDL).
The harmonized, filtered, PII-scrubbed derivative distributed in this deposit is released CC-BY-4.0;
the accompanying code is MIT.

# Test-A Failure Overlap

## Summary

| Failure definition | Final | SkillX | Intersection | Jaccard | Final covered | SkillX covered |
|---|---:|---:|---:|---:|---:|---:|
| Zero reward | 7 | 7 | 6 | 0.750 | 85.7% | 85.7% |
| Non-full reward | 46 | 51 | 44 | 0.830 | 95.7% | 86.3% |

Final/SkillX reward correlation is `0.878`.

## Common zero-reward tasks

| Sample | Goal | Final steps | SkillX steps |
|---|---|---:|---:|
| webshop:232 | i am interested in red heavy duty bed frames for a queen sized bed, and price lower than 330.00 dollars | 20 | 20 |
| webshop:492 | i'm looking for an original lightweight lace-up boot for my little toddler; she's a size 7, and price lower than 50.00 dollars | 20 | 20 |
| webshop:499 | i'm looking for furniture it was in living room  the color was pastel blue, and price lower than 80.00 dollars | 4 | 7 |
| webshop:664 | buy me a travel sized bottle of impression chanel 1932, and price lower than 40.00 dollars | 20 | 20 |
| webshop:951 | i need to order a game boy color. make sure that it's green and has stereo sound, and price lower than 60.00 dollars | 20 | 20 |
| webshop:969 | i'm looking for a cheese platter that i can include in a gift basket, and price lower than 80.00 dollars | 4 | 6 |

## Method-specific hard failures

### final_only_zero

- `webshop:87`: i need to order a fully assembled tan chair, and price lower than 120.00 dollars

### skillx_only_zero

- `webshop:407`: i would like a 120 wide by 90 long color 8 window panel that is machine washable, and price lower than 80.00 dollars

# Edit-category distribution

The 500 matched pairs are evenly balanced at 100 pairs per edit category. Each 100-pair batch contains exactly 20 of each category.

| Edit category | Pairs |
|---|---:|
| Amount | 100 |
| Recipient or merchant name | 100 |
| Date/time | 100 |
| Transaction ID/reference | 100 |
| Other supported field: fictional bank account suffix | 100 |
| **Total** | **500** |

The account-suffix edit is used only in the three transaction-detail families that visibly render that field. Recharge layouts use the other four categories. Machine-readable counts: `edit_category_distribution.json`.

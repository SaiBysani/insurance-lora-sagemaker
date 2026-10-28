# Side-by-side example (test policy 9)

Real outputs from the SageMaker AI run. Synthetic policy.

## Policy

```text
CRESCENT GENERAL
Homeowners Insurance Policy (HO-3)

DECLARATIONS
Policy Number: HO-322918-48
Named Insured: Aisha Rahman
Policy Period: 2026-06-09 to 2027-06-09, 12:01 a.m. standard time
Limit of Insurance: $150,000 per occurrence
Deductible: $500 per occurrence

SECTION I - GENERAL PROVISIONS
Premium shown in the Declarations was computed based on rates in effect at the time the policy was issued.
This policy may be cancelled by you at any time by written notice. We may cancel by giving you 30 days' written notice.

SECTION II - EXCLUSIONS
We will not pay for loss or damage caused directly or indirectly by any of the following:
A. Freezing of a plumbing system while the dwelling is vacant or unheated.
B. Earth movement of any kind, including sinkhole collapse and subsidence. However, if earth movement results in fire, we will pay for the loss caused by that fire.
C. The presence, growth, or spread of fungi, wet rot, or dry rot. This exclusion does not apply to mold caused by a covered water loss that you report to us within 14 days.

SECTION III - CONDITIONS
We will not pay for loss unless you have complied with all of the terms of this policy, including the duties after loss.
If there is other insurance covering the same loss, we will pay only our share of the loss.

CLAIM NOTICE
On 2027-04-14, tremors knocked over a space heater, which started a fire in the office. The insured is claiming $49,500.
```

## Base model answer

```text
{
  "policy_number": "HO-322918-48",
  "policy_type": "homeowners",
  "insured": "Aisha Rahman",
  "effective_date": "2026-06-09",
  "expiration_date": "2027-06-09",
  "coverage_limit_usd": 150000,
  "deductible_usd": 500,
  "exclusions": [
    {
      "peril": "freezing",
      "has_exception": false
    },
    {
      "peril": "earthquake",
      "has_exception": false
    },
    {
      "peril": "mold",
      "has_exception": true
    }
  ],
  "claim": {
    "peril": "fire",
    "decision": "covered",
    "payable_usd": 49000
  }
}
```

## Fine-tuned (LoRA) answer

```json
{"policy_number": "HO-322918-48", "policy_type": "homeowners", "insured": "Aisha Rahman", "effective_date": "2026-06-09", "expiration_date": "2027-06-09", "coverage_limit_usd": 150000, "deductible_usd": 500, "exclusions": [{"peril": "freezing", "has_exception": false}, {"peril": "earthquake", "has_exception": true}, {"peril": "mold", "has_exception": true}], "claim": {"peril": "earthquake", "decision": "covered_by_exception", "payable_usd": 49000}}
```

## Correct answer

```json
{
  "policy_number": "HO-322918-48",
  "policy_type": "homeowners",
  "insured": "Aisha Rahman",
  "effective_date": "2026-06-09",
  "expiration_date": "2027-06-09",
  "coverage_limit_usd": 150000,
  "deductible_usd": 500,
  "exclusions": [
    {
      "peril": "freezing",
      "has_exception": false
    },
    {
      "peril": "earthquake",
      "has_exception": true
    },
    {
      "peril": "mold",
      "has_exception": true
    }
  ],
  "claim": {
    "peril": "earthquake",
    "decision": "covered_by_exception",
    "payable_usd": 49000
  }
}
```

Score differences (base -> fine-tuned): exclusions_f1: 0.6666666666666666 -> 1.0, claim_peril: False -> True, decision: False -> True

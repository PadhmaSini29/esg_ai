import json
import os

with open(os.path.join("data", "esg_scores.json")) as f:
    scores = json.load(f)

def compare_companies(c1, c2):
    r1 = scores.get(c1, {})
    r2 = scores.get(c2, {})

    if not r1 or not r2:
        return "❌ One of the companies doesn't have ESG scores."

    result = f"📊 ESG Comparison: {c1} vs {c2}\n"
    for pillar in ["environment", "social", "governance"]:
        diff = r1[pillar] - r2[pillar]
        leader = c1 if diff > 0 else c2
        result += f"- {pillar.title()}: {leader} leads by {abs(diff)} points\n"
    
    result += f"✅ Total ESG Score — {c1}: {r1['score']} | {c2}: {r2['score']}"
    return result

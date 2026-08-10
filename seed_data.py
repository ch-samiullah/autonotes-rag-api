import requests

BASE_URL = "http://localhost:8000"

sample_documents = [
    {
        "source_name": "autonotes_features",
        "text": """AutoNotes Pro ek AI-powered note generation tool hai.
        Key Features:
        1. Real-time transcription
        2. Smart summarization
        3. Action item extraction
        4. Multi-language support — Urdu, Hindi, English
        5. Export options — PDF, Word, Notion, Slack
        Pricing:
        Free tier: 10 notes per month
        Pro: Rs. 999/month
        Enterprise: custom pricing"""
    },
    {
        "source_name": "company_hr_policy",
        "text": """HR Policy — AutoNotes Company
        Leave Policy:
        Annual leave: 21 days per year
        Sick leave: 10 days per year
        Maternity leave: 3 months fully paid
        Work Hours: 9 AM to 6 PM
        Remote work: 2 days per week allowed
        Benefits:
        Health insurance for employee and family
        Annual bonus: 1 month salary
        Learning budget: Rs. 50,000 per year"""
    },
    {
        "source_name": "technical_docs",
        "text": """AutoNotes API Documentation
        Authentication: Bearer token required
        Header: Authorization: Bearer YOUR_API_KEY
        Endpoints:
        POST /api/v1/notes/create
        GET /api/v1/notes/{id}
        GET /api/v1/notes
        DELETE /api/v1/notes/{id}
        Rate Limits:
        Free tier: 100 requests/day
        Pro tier: 10,000 requests/day
        Enterprise: unlimited"""
    }
]

def seed():
    print("Seeding knowledge base...")
    for doc in sample_documents:
        response = requests.post(f"{BASE_URL}/documents/add-text", json=doc)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ '{doc['source_name']}' → {data['chunks_created']} chunks added")
        else:
            print(f"✗ Error: {response.text}")
    stats = requests.get(f"{BASE_URL}/stats").json()
    print(f"\nTotal chunks in KB: {stats['total_chunks']}")
    print("Seeding complete!")

if __name__ == "__main__":
    seed()
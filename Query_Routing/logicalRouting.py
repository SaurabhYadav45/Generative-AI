def logical_routing(query):
    financial_keyword = ["finance", "financial", "stocks", "stock market", 
    "mutual fund", "mutual funds", "portfolio", "dividend", "ipo", "nifty",
    "sensex", "trading", "intraday", "equity", "debt", "demat", "share",
    "brokerage", "savings", "budget", "loan", "credit", "debit", "interest",]

    technical_keyword = ["python", "javascript", "java", "c++", "node", "react", "backend", "frontend","api", "database", "sql", "cloud", "aws", "devops", "docker",
    "git", "machine learning", "ai", "data science", "algorithm", "bug", "debug",
    "deployment", "code", "programming", "software", "framework", "architecture"]

    hr_keyword= ["recruitment", "hiring", "interview", "candidate", "onboarding", 
    "cv", "offer letter", "employee", "team", "hr", "appraisal", "benefits",
    "salary", "promotion", "policy", "work culture", "leave", "payroll", "resume",
    "compliance", "training", "performance", "feedback", "talent acquisition"]

    if any(word in query.lower() for word in financial_keyword):
        return "Financial_docs"
    elif any(word in query.lower() for word in technical_keyword):
        return "Technical_docs"
    elif any(word in query.lower() for word in hr_keyword):
        return "HR_docs"
    else:
        return "General_docs"
    
query = input("Ask you query: ")
selected_docs = logical_routing(query)

print("Selected Docs :", selected_docs)
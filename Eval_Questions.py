QUESTIONS = [

    #### Q 9 and 10 are both left unanswerable to fill project requirements for 2 unanswerable questions
    ### Both Q 9 and 10 will not have relevant id from the corpus to satisfy unanswerable questions

    {
        "id": "Q1",
        "type": "factoid",
        "question": "How much did the CN freight train involved in the Longueuil, Quebec derailment on November 14, 2024 weigh?",
        "reference_answer": "The train weighed approximately 9,156 tons.",
        "relevant": ["256_0"]
    },

    {
        "id": "Q2",
        "type": "factoid",
        "question": "What happened to the CN freight train involved in the rail occurrence at St. Catharines, Ontario on March 30, 2026?",
        "reference_answer": "The train derailed near Bridge 6 and 13 railcars left the track.",
        "relevant": ["39_0"]
    },

    {
        "id": "Q3",
        "type": "factoid",
        "question": "Where did Canadian National freight train M-365-21-23 derail?",
        "reference_answer": "The derailment occurred near Clova, Quebec at Mile 165.80 of the Saint-Maurice Subdivision.",
        "relevant": ["7_0"]
    },

    {
        "id": "Q4",
        "type": "factoid",
        "question": "How many cars derailed in the occurrence involving Canadian National freight train M-365-21-23?",
        "reference_answer": "Seventeen railcars derailed.",
        "relevant": ["7_0"]
    },

    {
        "id": "Q5",
        "type": "factoid",
        "question": "Was there any permanent environmental damage in the derailment of Canadian National freight train M-365-21-23?",
        "reference_answer": "No permanent environmental damage was reported.",
        "relevant": ["7_0"]
    },

    {
        "id": "Q6",
        "type": "factoid",
        "question": "Approximately how much did Canadian National freight train M-365-21-23 weigh?",
        "reference_answer": "The train weighed approximately 13,100 tons.",
        "relevant": ["7_0"]
    },

    {
        "id": "Q7",
        "type": "multi-hop",
        "question": "Compare the Longueuil, Quebec derailment and the St. Catharines, Ontario derailment. What happened in each occurrence?",
        "reference_answer": "The Longueuil occurrence involved a CN freight train derailment in Longueuil, Quebec, where 8 cars derailed. The St. Catharines occurrence involved a CN freight train derailment at Bridge 6 in St. Catharines, Ontario, where 13 cars derailed.",
        "relevant": ["256_0", "39_0"]
    },

    {
        "id": "Q8",
        "type": "multi-hop",
        "question": "Compare the Longueuil, Quebec derailment and the Clova, Quebec derailment. Which train was involved in each occurrence?",
        "reference_answer": "The Longueuil occurrence involved Canadian National freight train M32231-13 in Longueuil, Quebec. The Clova occurrence involved Canadian National freight train M-365-21-23 near Clova, Quebec.",
        "relevant": ["256_0", "7_0"]
    },

    {
        "id": "Q9",
        "type": "unanswerable",
        "question": "What did the CN train that derailed on November 15, 2024 in Quebec weigh?",
        "reference_answer": "I don't know.",
        "relevant": []
    },

    {
        "id": "Q10",
        "type": "unanswerable",
        "question": "Which Transportation Safety Board report describes the derailment of a VIA Rail passenger train in Whitehorse, Yukon during 2025?",
        "reference_answer": "I don't know.",
        "relevant": []
    }

]
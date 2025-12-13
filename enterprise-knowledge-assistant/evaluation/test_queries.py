"""Test queries for RAG evaluation"""

# SAP Knowledge Base Test Queries
SAP_TEST_QUERIES = [
    {
        'question': 'What is SAP\'s policy on accepting gifts?',
        'expected_doc': 'Global_CoE_BizConduct.pdf',
        'expected_topic': 'gifts and entertainment policy'
    },
    {
        'question': 'What are the AI ethics principles at SAP?',
        'expected_doc': 'Global_AI_Ethics_Policy.pdf',
        'expected_topic': 'AI ethical principles'
    },
    {
        'question': 'What is the AI Ethics Steering Committee?',
        'expected_doc': 'Global_AI_Ethics_Policy.pdf',
        'expected_topic': 'governance structure'
    },
    {
        'question': 'Can partners accept cash payments from customers?',
        'expected_doc': 'SAP_Partner_CoC.pdf',
        'expected_topic': 'anti-corruption policy'
    },
    {
        'question': 'What are SAP\'s transparency requirements for AI?',
        'expected_doc': 'Global_AI_Ethics_Policy.pdf',
        'expected_topic': 'transparency and explainability'
    }
]

# Salesforce Knowledge Base Test Queries
SALESFORCE_TEST_QUERIES = [
    {
        'question': 'What is Salesforce Apex?',
        'expected_doc': 'salesforce_apex_language_reference.pdf',
        'expected_topic': 'Apex programming language'
    },
    {
        'question': 'How do SOQL queries work?',
        'expected_doc': 'salesforce_soql_sosl.pdf',
        'expected_topic': 'SOQL query syntax'
    },
    {
        'question': 'What are governor limits in Salesforce?',
        'expected_doc': 'salesforce_apex_language_reference.pdf',
        'expected_topic': 'execution limits'
    }
]
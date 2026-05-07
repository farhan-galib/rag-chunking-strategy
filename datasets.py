"""
Sample datasets for testing various chunking strategies
"""

# Dataset 1: Technical Article (FAQ-style)
TECHNICAL_FAQ = """
# Machine Learning Basics

## What is Machine Learning?

Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming. 
It focuses on developing algorithms that can analyze data, identify patterns, and make decisions with minimal human intervention.

## Types of Machine Learning

### Supervised Learning

Supervised learning is a machine learning paradigm where the training data includes both input features and their corresponding target outputs. 
The algorithm learns to map inputs to outputs by minimizing prediction errors. Common applications include regression and classification tasks.

Examples include linear regression for predicting continuous values and decision trees for classification problems.

### Unsupervised Learning

Unsupervised learning works with data that has no labeled outputs. The algorithm discovers hidden patterns and structures in the data. 
Common techniques include clustering and dimensionality reduction.

K-means clustering is a popular unsupervised algorithm for grouping similar data points together.

### Reinforcement Learning

Reinforcement learning involves an agent learning to make decisions by interacting with an environment. 
The agent receives rewards or penalties for its actions and learns to maximize cumulative rewards over time.

Applications include game playing, robotics, and autonomous systems.

## Neural Networks Explained

Neural networks are inspired by biological neurons in the human brain. They consist of interconnected layers of artificial neurons that process information.

The input layer receives raw data, hidden layers process information, and the output layer produces predictions.

Training neural networks involves forward propagation to make predictions and backpropagation to update weights based on errors.

## Deep Learning

Deep learning uses neural networks with many layers (hence "deep") to learn hierarchical representations of data.

It has achieved remarkable success in computer vision, natural language processing, and speech recognition tasks.

Popular architectures include Convolutional Neural Networks (CNNs) for image processing and Recurrent Neural Networks (RNNs) for sequence data.
"""

# Dataset 2: Legal Document (Long, structured)
LEGAL_DOCUMENT = """
# TERMS OF SERVICE AGREEMENT

## 1. DEFINITIONS AND INTERPRETATION

1.1 In this Agreement, unless the context otherwise requires, the following terms shall have the meanings respectively assigned to them:

"Agreement" means this terms of service agreement including all schedules and appendices, as amended from time to time.

"Service" means the online platform and related services provided by the Company to Users.

"User" means any individual or entity that accesses or uses the Service.

"Content" means any information, data, materials, or communications provided by Users or third parties on the Service.

1.2 The singular includes the plural and vice versa. Words importing a person include a firm and a body corporate.

## 2. ACCEPTANCE OF TERMS

2.1 By accessing and using the Service, you acknowledge that you have read, understood, and agree to be bound by this Agreement.

2.2 If you do not agree to be bound by this Agreement, you may not access or use the Service.

2.3 The Company reserves the right to modify this Agreement at any time, and such modifications shall be effective upon posting to the Service.

## 3. USER RESPONSIBILITIES

3.1 Users are responsible for maintaining the confidentiality of their account credentials and for all activities conducted under their account.

3.2 Users agree not to engage in any conduct that could interfere with or impair the function of the Service.

3.3 Users shall not upload, post, or transmit any content that is unlawful, defamatory, obscene, or otherwise objectionable.

3.4 Users grant the Company a non-exclusive, worldwide license to use, reproduce, and display User-submitted Content.

## 4. LIMITATION OF LIABILITY

4.1 In no event shall the Company be liable for any indirect, incidental, special, or consequential damages arising out of or in connection with this Agreement or the Service.

4.2 The Company's total liability shall not exceed the amount paid by the User in the twelve months preceding the claim.

## 5. DISPUTE RESOLUTION

5.1 Any disputes arising under this Agreement shall be governed by the laws of the jurisdiction in which the Company is located.

5.2 Both parties agree to attempt to resolve disputes through good faith negotiation before pursuing legal action.

5.3 If negotiation fails, disputes shall be resolved through binding arbitration.
"""

# Dataset 3: Medical Research Paper
MEDICAL_RESEARCH = """
# Clinical Efficacy of Novel Immunotherapy in Advanced Melanoma Patients

## Abstract

Melanoma remains one of the most aggressive malignancies, with increasing incidence rates globally. 
This study evaluates the clinical efficacy and safety profile of a novel immunotherapy combination in 150 patients with advanced melanoma.

## Introduction

Melanoma accounts for approximately 1-2% of all skin cancers but is responsible for the majority of skin cancer-related deaths. 
Recent advances in immunotherapy have significantly improved survival outcomes in advanced melanoma patients.

The programmed cell death protein 1 (PD-1) inhibitors have demonstrated remarkable clinical benefit in melanoma treatment. 
However, resistance to PD-1 inhibition occurs in approximately 30-40% of patients.

This study aims to evaluate whether combining PD-1 inhibitors with lymphocyte-activating gene 3 (LAG-3) inhibitors can improve therapeutic outcomes.

## Methods

This was a randomized, double-blind, Phase III clinical trial conducted at 25 medical centers across North America and Europe.

Eligible patients had histologically confirmed stage III or IV melanoma and had not received prior systemic therapy.

Patients were randomized 1:1 to receive either PD-1 inhibitor monotherapy or combination therapy with PD-1 and LAG-3 inhibitors.

The primary endpoint was overall survival (OS) at 24 months. Secondary endpoints included progression-free survival (PFS), objective response rate (ORR), and safety profiles.

## Results

Among 150 patients enrolled, 75 received PD-1 monotherapy and 75 received combination therapy.

Median overall survival was 24.5 months in the combination group versus 18.3 months in the monotherapy group (HR 0.72, 95% CI 0.58-0.89, p=0.002).

The objective response rate was 52% in the combination group versus 38% in the monotherapy group.

Adverse events were comparable between groups, with immune-related adverse events occurring in 45% of combination therapy patients versus 38% in monotherapy patients.

## Discussion

This study demonstrates that combination immunotherapy provides superior survival benefit compared to monotherapy in advanced melanoma.

The mechanism underlying improved efficacy likely involves enhanced T-cell activation through dual checkpoint inhibition.

Further research is warranted to identify predictive biomarkers for patient selection and to evaluate this combination in other malignancies.

## Conclusion

Combination PD-1 and LAG-3 inhibition represents a promising therapeutic strategy for advanced melanoma and warrants consideration as a new standard of care.
"""

# Dataset 4: Code Documentation
CODE_DOCUMENTATION = """
# API Documentation

## Authentication

The API uses OAuth 2.0 for authentication. All requests must include a valid access token in the Authorization header.

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

To obtain an access token, send a POST request to the /oauth/token endpoint with your client credentials.

## Endpoints

### GET /api/v1/users/:id

Retrieve user information by user ID.

**Parameters:**
- id (required): The unique identifier of the user
- include_metadata (optional): Include additional user metadata (boolean)

**Response:**
```json
{
  "id": "user123",
  "name": "John Doe",
  "email": "john@example.com",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### POST /api/v1/users

Create a new user account.

**Request Body:**
```json
{
  "name": "Jane Smith",
  "email": "jane@example.com",
  "password": "secure_password"
}
```

**Response:**
Returns the created user object with HTTP status 201.

### DELETE /api/v1/users/:id

Delete a user account.

**Parameters:**
- id (required): The unique identifier of the user to delete

**Response:**
Returns HTTP status 204 on success.

## Error Handling

The API returns standard HTTP status codes and error messages in the following format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request parameters are invalid"
  }
}
```

Common error codes include:
- INVALID_REQUEST: The request parameters are malformed
- UNAUTHORIZED: Authentication failed or token is invalid
- FORBIDDEN: User does not have permission for this resource
- NOT_FOUND: The requested resource does not exist
- RATE_LIMIT_EXCEEDED: Too many requests sent in a short time period

## Rate Limiting

The API enforces rate limiting at 1000 requests per hour per API key.
Rate limit information is included in response headers:

- X-RateLimit-Limit: Total requests allowed
- X-RateLimit-Remaining: Requests remaining in current window
- X-RateLimit-Reset: Unix timestamp when the limit resets
"""

# Dataset 5: E-commerce Product Descriptions
ECOMMERCE_PRODUCTS = """
# Product Catalog

## Product 1: Wireless Noise-Cancelling Headphones

### Overview

Experience premium audio quality with our flagship wireless noise-cancelling headphones. 
Engineered for comfort and performance, these headphones deliver immersive sound for music enthusiasts and professionals alike.

### Key Features

- Active Noise Cancellation: Advanced noise cancellation technology reduces ambient noise by up to 30dB
- Bluetooth 5.0: Seamless wireless connectivity up to 30 meters
- Battery Life: Up to 40 hours of continuous playback on a single charge
- Comfort Design: Ergonomic padding and adjustable headband for extended wear
- Premium Drivers: Custom-tuned 40mm drivers deliver rich, balanced sound

### Technical Specifications

- Frequency Response: 20Hz - 20kHz
- Impedance: 32 Ohms
- Sensitivity: 105dB SPL at 1kHz
- Bluetooth Version: 5.0
- Codecs Supported: AAC, SBC, aptX, LDAC

### What's in the Box

- Headphones with detachable cable
- USB-C charging cable
- 3.5mm audio cable
- Carrying case
- User manual and warranty information

### Customer Support

We offer 2-year manufacturer's warranty and 30-day money-back guarantee. 
Customer support is available 24/7 via email, phone, and live chat.

## Product 2: Portable Solar Power Bank

### Overview

The ultimate portable charging solution for outdoor enthusiasts and travelers. 
This solar-powered power bank harnesses the sun's energy to keep your devices charged anywhere.

### Key Features

- Solar Panel: High-efficiency 10W solar panel charges in sunlight
- Capacity: 25,000mAh battery capacity
- Dual USB Ports: Charge two devices simultaneously
- Fast Charging: Quick Charge 3.0 support for rapid charging
- Rugged Design: Water-resistant, dustproof, and shockproof
- LED Indicator: Battery level display with LED flashlight

### Specifications

- Battery Type: Lithium Polymer
- Solar Charging Time: 30-40 hours (depending on sunlight)
- USB Charging Time: 8-10 hours
- Output: Dual USB 5V/2.4A
- Dimensions: 150 x 85 x 20mm
- Weight: 450g

### Warranty

Covered by 1-year manufacturer's warranty against defects in materials and workmanship.
"""

# Dataset 6: Customer Support Conversations
SUPPORT_CONVERSATIONS = """
# Customer Support Ticket #12345

## Ticket Summary
Customer unable to reset password and locked out of account.

## Conversation History

**Customer (2024-05-07 10:15 AM):**
I'm unable to reset my password. I've tried clicking the "Forgot Password" link multiple times, but I'm not receiving the reset email. 
This is really urgent as I need to access my account today for an important transaction.

**Support Agent - Sarah (2024-05-07 10:30 AM):**
Thank you for contacting us. I understand how frustrating this situation must be. Let me help you resolve this issue.

First, could you please confirm:
1. The email address associated with your account
2. Whether you've checked your spam/junk folder for the reset email
3. If you're using a corporate email or Gmail/Outlook

**Customer (2024-05-07 10:45 AM):**
The email is john.smith@company.com (corporate email). Yes, I've checked spam already. No reset email appears anywhere.

**Support Agent - Sarah (2024-05-07 11:00 AM):**
Thank you for confirming those details. Corporate email filters can sometimes block our password reset emails. 

I've manually reset your password and sent you a new reset link to john.smith@company.com. 
Please check your email within the next 5 minutes and follow the reset link. 

If you don't see the email, please:
1. Contact your IT department to check email filtering rules
2. Ask them to whitelist emails from reset@ourservice.com

Please let me know once you've successfully reset your password.

**Customer (2024-05-07 11:15 AM):**
Great! I received the email and successfully reset my password. I'm now able to access my account. Thank you so much for the quick help!

**Support Agent - Sarah (2024-05-07 11:20 AM):**
Excellent! I'm glad we could resolve this for you. To prevent this issue in the future, consider adding our domain to your corporate email's whitelist.

If you encounter any other issues, please don't hesitate to reach out. We're here to help!

**Ticket Status: RESOLVED**
"""

# Consolidated dataset dictionary
DATASETS = {
    'technical_faq': TECHNICAL_FAQ,
    'legal_document': LEGAL_DOCUMENT,
    'medical_research': MEDICAL_RESEARCH,
    'code_documentation': CODE_DOCUMENTATION,
    'ecommerce_products': ECOMMERCE_PRODUCTS,
    'support_conversations': SUPPORT_CONVERSATIONS
}

def get_dataset(name: str) -> str:
    """Retrieve a dataset by name"""
    return DATASETS.get(name, "")

def list_datasets():
    """List all available datasets"""
    return list(DATASETS.keys())
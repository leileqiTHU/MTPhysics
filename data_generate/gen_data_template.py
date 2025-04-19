import random
import json
import jsonlines
from faker import Faker
from tqdm import trange
import os

# Initialize Faker for generating fake data
fake = Faker()

# Load the templates from the JSON file
with open('templates.json', 'r', encoding='utf-8') as f:
    templates = json.load(f)

# Define static lists for specific fields to ensure realism
universities = [
    "Massachusetts Institute of Technology",
    "Stanford University",
    "Harvard University",
    "University of California, Berkeley",
    "California Institute of Technology",
    "Princeton University",
    "University of Oxford",
    "University of Cambridge",
    "ETH Zurich",
    "Imperial College London",
    "University of Chicago",
    "Columbia University",
    "University of Pennsylvania",
    "Yale University",
    "University of Toronto",
    "University of Tokyo",
    "National University of Singapore",
    "University of Melbourne",
    "Peking University",
    "Tsinghua University",
    "University of Edinburgh",
    "University of Michigan",
    "University of Hong Kong",
    "University of Texas at Austin",
    "University of California, Los Angeles",
    "Johns Hopkins University",
    "University of Washington",
    "Duke University",
    "University of Illinois at Urbana-Champaign",
    "University of Wisconsin-Madison",
    "University of British Columbia",
    "New York University",
    "University of Manchester",
    "Carnegie Mellon University",
    "University of Sydney",
    "University of Amsterdam",
    "Seoul National University",
    "McGill University",
    "University of California, San Diego",
    "University of Minnesota",
    "University of Bristol",
    "University of Queensland",
    "University of Warwick",
    "University of Copenhagen",
    "University of Glasgow",
    "Kyoto University",
    "University of Zurich",
    "KU Leuven",
    "University of Auckland",
    "Boston University",
    "University of Dublin",
    "Leiden University"
]

fields_of_study = [
    "Communications",
    "Computer Science",
    "Mechanical Engineering",
    "Electrical Engineering",
    "Biotechnology",
    "Physics",
    "Mathematics",
    "Economics",
    "Psychology",
    "Sociology",
    "Chemistry",
    "Biology",
    "Civil Engineering",
    "Environmental Science",
    "Political Science",
    "Business Administration",
    "Law",
    "Medicine",
    "Philosophy",
    "History",
    "Art History",
    "Literature",
    "Fine Arts",
    "Graphic Design",
    "Architecture",
    "Astronomy",
    "Statistics",
    "Information Technology",
    "Data Science",
    "Nursing",
    "Public Health",
    "Marketing",
    "Finance",
    "Accounting",
    "International Relations",
    "Anthropology",
    "Geography",
    "Linguistics",
    "Education",
    "Theater Arts",
    "Music",
    "Film Studies",
    "Creative Writing",
    "Journalism",
    "Econometrics",
    "Operations Research",
    "Biomedical Engineering",
    "Neuroscience",
    "Cognitive Science",
    "Human Resource Management",
    "Public Policy"
]

companies = [
    "Meta Platforms",
    "Google",
    "Microsoft",
    "Apple",
    "Amazon",
    "Tesla",
    "IBM",
    "Intel",
    "Cisco Systems",
    "Oracle",
    "Facebook",
    "Netflix",
    "Adobe",
    "Salesforce",
    "Uber",
    "Airbnb",
    "Twitter",
    "Spotify",
    "Dropbox",
    "PayPal",
    "Shopify",
    "Zoom Video Communications",
    "Square",
    "Snap Inc.",
    "Slack Technologies",
    "Lyft",
    "Pinterest",
    "Red Hat",
    "Qualcomm",
    "VMware",
    "NVIDIA",
    "Stripe",
    "Palantir Technologies",
    "HubSpot",
    "ServiceNow",
    "Workday",
    "Intuit",
    "Atlassian",
    "DocuSign",
    "eBay",
    "SAP",
    "Siemens",
    "Accenture",
    "Dell Technologies",
    "HP Inc.",
    "Capgemini",
    "Infosys",
    "Tata Consultancy Services",
    "Wipro",
    "Cognizant",
    "HCL Technologies",
    "Mindtree",
    "L&T Infotech",
    "Baidu",
    "ByteDance",
    "Huawei Technologies",
    "Tencent",
    "Alibaba Group",
    "JD.com",
    "Lenovo",
    "Asus",
    "Nokia",
    "Philips",
    "Panasonic",
    "LG Electronics",
    "Sony",
    "Samsung Electronics",
    "Hitachi",
    "Fujitsu",
    "Canon",
    "Sharp",
    "Xiaomi",
    "ZTE",
    "Toshiba",
    "NEC",
    "Motorola",
    "BlackBerry",
    "Nikon",
    "Ricoh",
    "Seagate Technology",
    "Western Digital",
    "Micron Technology",
    "Texas Instruments",
    "Broadcom",
    "Advanced Micro Devices (AMD)"
]

# Function to generate a fake biography profile
def generate_profile():
    birth_date = fake.date_of_birth(tzinfo=None, minimum_age=18, maximum_age=90)
    profile = {
        "name": fake.name(),
        "birth_date": birth_date.strftime("%B %d, %Y"),
        "early_city": fake.city(),
        "institution": random.choice(universities),
        "field": random.choice(fields_of_study),
        "company": random.choice(companies),
        "employ_loc": fake.city(),
    }
    return profile

# Function to generate a set of questions and answers based on a profile
def generate_qa(profile):
    questions = [
        f"When was {profile['name']} born?",
        f"Where did {profile['name']} spend their early years?",
        f"Where did {profile['name']} receive mentorship and guidance?",
        f"Which field did {profile['name']} focus on during their education?",
        f"Which company was {profile['name']} employed by?",
        f"Which city was {profile['name']} employed in?"
    ]
    
    answers = [
        profile["birth_date"],
        profile["early_city"],
        profile["institution"],
        profile["field"],
        profile["company"],
        profile["employ_loc"]
    ]
    
    full_texts = [f"{q} Answer: {a}" for q, a in zip(questions, answers)]
    
    return {
        "questions": questions,
        "answers": answers,
        "full_texts": full_texts
    }

# Function to generate a biography using templates
def generate_bio(profile, templates):
    bio_sentences = []
    
    # Define the mapping between template categories and profile data
    template_mapping = {
        "birth": {
            "NAME": profile["name"],
            "BIRTHDAY": profile["birth_date"]
        },
        "early_years": {
            "NAME": profile["name"],
            "LOCATION": profile["early_city"]
        },
        "mentorship": {
            "NAME": profile["name"],
            "INSTITUTION": profile["institution"]
        },
        "education_focus": {
            "NAME": profile["name"],
            "FIELD": profile["field"]
        },
        "professional_role": {
            "NAME": profile["name"],
            "COMPANY": profile["company"]
        },
        "employment_location": {
            "NAME": profile["name"],
            "EMPLOY_LOC": profile["employ_loc"],
        }
    }
    
    # Iterate through each template category and select a random variant
    for category, placeholders in template_mapping.items():
        chosen_template = random.choice(templates[category])
        try:
            sentence = chosen_template.format(**placeholders)
        except KeyError as e:
            # Handle missing placeholders gracefully
            print(f"Missing placeholder {e} in category '{category}'.")
            sentence = ""
        bio_sentences.append(sentence)
    
    # Combine all sentences to form the full biography
    bio = " ".join(filter(None, bio_sentences))  # Filter out any empty sentences
    return bio

# Path to save the generated data
output_file = './data_gen/bio_with_qas_repaired_template.jsonl'

# Total number of bios to generate
total_num = 100000
batch_size = 100  # Increased batch size for efficiency
num_batches = total_num // batch_size

# Ensure the output directory exists
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Initialize JSON Lines writer
with jsonlines.open(output_file, mode='a') as writer:
    for _ in trange(num_batches, desc="Generating Bios"):
        batch_gen = []
        for _ in range(batch_size):
            profile = generate_profile()
            qa = generate_qa(profile)
            bio = generate_bio(profile, templates)
            
            batch_gen.append({
                "questions": qa["questions"],
                "answers": qa["answers"],
                "full_texts": qa["full_texts"],
                "bio": bio
            })
        
        # Write the entire batch at once for efficiency
        writer.write_all(batch_gen)

print(f"Data generation complete. Bios saved to '{output_file}'.")

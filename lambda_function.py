import json
import boto3
import requests
from dotenv import load_dotenv
import os

load_dotenv()

bedrock = boto3.client("bedrock-runtime") 


def generate_prompt(company_name,  receiver_name, success_stories, services_list,sender_name,company_info):
    prompt = f"""
            You are a cold email copywriter specialized in B2B personalization using the “Show Me You Know Me” method by Sam McKenna. 
            Based on the structured research data below (from Perplexity), write a highly personalized and relevant cold email that follows this exact structure: 

            Important: 75% of emails are read using a phone

            Email Subject: Specific, personal, relevant only to this person. No generic hooks. Must feel personal and unique. Use a detail that could trigger interest in the prospect.

            ⚠️ Important: The content of the email must be composed by:

                ### First sentence / preview text (Start with direct reference to a human detail from the data. Show empathy and familiarity and a question whatever they're having some ideas to improve their process but don't know from where to start)  ###

                ### Value proposition (Clearly state a relevant problem you can solve for this person’s role, company or industry. 
                # Cross check the pain points for similarities between the data from Perplexity and the pain points below. 
                # Determine how you can solve it by using the most relatable and appropriate solution. 
                # Still list and show the other solutions too for reference.
                # Here are our solutions with short descriptions and related pain point
                #
                Our Solution we Offer:
                - Intelligent document processing: Process any incoming document with less than 0.01 USD per page and high accuracy (>90%)
                - AI Voice Agents for customer support, appointment scheduling (at 0.01/2 USD per minute)
                - AI Chatbots: Smart chat with knowledge of your data!
                - Digital transformation: Automate and optimize business processes with AI-driven solutions (Power Automate, RPA, Model Driven Apps)
                - Process Automation: Streamline workflows and reduce manual tasks with AI.
                - Knoowledge Management: Create knowledge bases to chat with more than 20k documents about technical documentation of products and projects.
                For all solutions, we offer a free demonstration and consultation to analyze your processes or needs and propose a solution.
                Then anticipate and handle the most likely objection.
                ###

                ### Closing: Be polite. Do not insert a calendar link. Use this sentence instead:
                “Would you have time in the next
                Use a human, conversational tone. Use the industry terminology. Do not write like a marketing bot. Your only goal is to get a reply. Do not insert asterisks or other markdown formatting.



        These are the prospect data:
            ### Company Name: {company_name}

            ### Company data and analysis
            {company_info}.

            ### Receiver Name
            {receiver_name}
            
            ### Our Services/Products
            {services_list}
            
            ### Success Stories
            {success_stories}
            
            
            Your job is to create a short form cold email using the company analysis (focusing on pain points) for the CEO of the company using the details and specific information making the email more personalized for the CEO.
            Example output:
            
            Hi [RECEIVER_NAME],

            [HOOK SHORT SENTENCE]

            [DESCRIPTION_OF_COMPANY_SPECIFIC_PROBLEMS (1 max 2)]
            
            [DESCRIPTION OF POSSIBLE SOLUTIONS BASED ON OUR SERVICES]

            [REFERENCE TO PREVIOUS SUCCESS STORIES]

            [CALL TO ACTION TO BOOK CALL]
            
            Best regards {sender_name}.
            ### Example output:

            {{
                'email_content':'content of the email',
                'email_subject':"subject"
            }}
            """

    return prompt

def get_company_info(company_url, company_description,vendor_description):
    print(f"Retrieving company info {company_url}")
    perplexity_url ='https://api.perplexity.ai/chat/completions'
    X_API_KEY = os.getenv("X_API_KEY")
    payload = {
        "temperature": 0.2,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "web_search_options": {"search_context_size": "low"},
        "model": "sonar",
        "messages": [
            {
                "role": "user",
                "content": f"Make a reserach about this company {company_url} ({company_description}) and give description about their business and most common pain points they might face right now that might be solved with our services/products provided by our company ({vendor_description})."
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {X_API_KEY}",
        "Content-Type": "application/json"
    }

    perplexity_response = requests.post(perplexity_url,json=payload,headers=headers)
    print(f"Company data retrieved")
    output_text = json.loads(perplexity_response.content.decode("utf-8"))
    output_text = output_text['choices'][0]['message']['content']
    print(output_text)
    return output_text


def lambda_handler(event, context):
    print(event)
    body = json.loads(event['body'])
    company_name = body['companyName']
    company_url = body['companyUrl'],
    company_description = body['companyDescription']
    receiver_name = body['receiverName']
    success_stories = body['successStories']
    services_list = body['servicesList']
    vendor_description = body['vendorDescription']
    sender_name = body['senderName']
    company_analysis = get_company_info(company_url, company_description, vendor_description)
    prompt = generate_prompt(company_name, receiver_name, success_stories, services_list, sender_name, company_analysis)


    prompt = f"{prompt}. Return only a valid JSON object as requested. Do not add anything else."
    payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 3000,
            "messages": [
                
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]},
                {"role": "assistant", "content": "Here is your JSON data without additional text before of after:"}
            ]
        }
    ai_response = bedrock.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        body=json.dumps(payload)
    )
    
    

    ai_output = json.loads(ai_response["body"].read().decode())
    ai_output = ai_output["content"][0]['text']
    output = parse_output(ai_output)
    
    email_content = output['email_content']
    email_subject = output['email_subject']
   

    return {
        'statusCode': 200,
        'body': json.dumps({"email_content":email_content,"email_subject":email_subject})
    }


def parse_output(output):

    try:
        # Strip everything except the JSON block
        start = output.index("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        
        # Optional: print cleaned JSON string
        print("Extracted JSON:\n", json_str)
        
        # Parse the JSON
        parsed = json.loads(json_str)
        return parsed

    except (ValueError, json.JSONDecodeError) as e:
        print("Failed to parse JSON:", e)
        return None


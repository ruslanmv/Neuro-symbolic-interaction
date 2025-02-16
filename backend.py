import os
import re
from getpass import getpass
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from owlready2 import *
from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import os
from dotenv import load_dotenv
default=True
# Check if in Google Colab
try:
    import google.colab
    in_colab = True
except ImportError:
    in_colab = False

if in_colab:
    # Get Watsonx credentials from Google Colab userdata
    from google.colab import userdata
    api_key = userdata.get('WATSONX_API_KEY')
    project_id = userdata.get('PROJECT_ID')
    url = userdata.get('WATSONX_URL')

    # Java path in Colab (no action needed, assumed to be set)
else: # Not in Google Colab
    load_dotenv() # Load from .env file

    # Get Watsonx credentials from .env file
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("PROJECT_ID")
    url = os.getenv("WATSONX_URL")

    # Java Path handling outside Colab
    if os.name == 'nt': # Check if Windows
        java_path = r'C:\Program Files\Common Files\Oracle\Java\javapath'
        os.environ['PATH'] = java_path + ';' + os.environ.get('PATH', '') # Prepend and ensure existing PATH is kept
    # else: # Not Windows (e.g., Linux, macOS) - assume standard Java path is sufficient
# Function to set environment variables
def set_env(var: str):
    env_var = os.getenv(var)
    if not env_var:
        env_var = getpass(f"{var}: ")
        os.environ[var] = env_var
    return env_var
# Define IBM connection parameters
class IbmConnectionParams(BaseModel):
    api_key: str
    project_id: str
    url: str
    credentials: dict[str, str]

    def __init__(self, api_key: str, project_id: str, url: str) -> None:
        super().__init__(api_key=api_key, project_id=project_id, url=url, credentials={"url": url, "apikey": api_key})

# Define parameters for the model
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 4095,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

# Initialize the WatsonxLLM model
watsonx_llm = WatsonxLLM(
    #model_id="meta-llama/llama-3-70b-instruct",
    model_id="meta-llama/llama-3-1-70b-instruct",
    apikey=api_key,
    url=url,
    project_id=project_id,
    params=parameters,
)

# Define the system prompt
system_prompt = (
    "You are an AI language model, designed to assist with a variety of tasks. "
    "You can provide detailed, accurate, and helpful responses on a wide range of topics, including answering questions, "
    "generating text, and offering explanations. Your goal is to assist users by providing the most relevant and useful information. "
    "When uncertain, it is better to acknowledge the uncertainty than to provide incorrect information."
)

# Initialize the prompt template
prompt_template = PromptTemplate(input_variables=[], template=system_prompt)

# Combine the system prompt with the user's prompt
def create_full_prompt(user_prompt: str) -> str:
    return f"{system_prompt}\n\n{user_prompt}"

# Function to interact with WatsonxLLM model
def ask_watsonx(user_prompt: str) -> str:
    response = watsonx_llm.invoke(create_full_prompt(user_prompt))
    return response

# Define ontology
from owlready2 import *
if default:
    # Define ontology
    onto = get_ontology("http://example.org/engine_ontology.owl")
    with onto:
        class Engine(Thing):
            pass

        class OilEngine(Engine):
            pass

        class ElectricEngine(Engine):
            pass

        class EngineComponent(Thing):
            pass

        class OilEngineComponent(EngineComponent):
            pass

        class ElectricEngineComponent(EngineComponent):
            pass

        class Piston(OilEngineComponent):
            pass

        class OilPump(OilEngineComponent):
            pass

        class Battery(ElectricEngineComponent):
            pass

        class Motor(ElectricEngineComponent):
            pass

        # Fix: CausesFailure should be an ObjectProperty
        class CausesFailure(ObjectProperty):
            domain = [EngineComponent]
            range = [Engine]

        # Instances
        piston = Piston("piston_1")
        oil_pump = OilPump("oil_pump_1")
        battery = Battery("battery_1")
        motor = Motor("motor_1")
        oil_engine = OilEngine("oil_engine_1")
        electric_engine = ElectricEngine("electric_engine_1")

        # Defining relationships
        piston.CausesFailure.append(oil_engine)
        oil_pump.CausesFailure.append(oil_engine)
        battery.CausesFailure.append(electric_engine)
        motor.CausesFailure.append(electric_engine)

    # Save the ontology
    onto.save(file="engine_ontology.owl")
    # Run the reasoner to infer new knowledge
    with onto:
        sync_reasoner()

from owlready2 import *

# Function to extract ontology information
def get_ontology_info(onto):
    info = "Ontology Information:\n"

    # Classes and Subclasses with Descriptions
    info += "\nClasses and Subclasses:\n"
    processed_subclasses = set()  # To avoid repetition
    for cls in onto.classes():
        info += f"- {cls.name} class:\n"
        if cls.comment:
            info += f"  - Description: {cls.comment[0]}\n"  # Include class description if available
        for subcls in cls.subclasses():
            if subcls not in processed_subclasses:  # Avoid printing duplicates
                info += f"  - Subclass: {subcls.name}\n"
                if subcls.comment:
                    info += f"    - Description: {subcls.comment[0]}\n"
                processed_subclasses.add(subcls)

    # Instances with Types and Descriptions
    info += "\nInstances:\n"
    for instance in onto.individuals():
        info += f"- {instance.name}: Instance of {instance.is_a[0].name}\n"
        if instance.comment:
            info += f"  - Description: {instance.comment[0]}\n" 

    # Object Properties (Relationships) without duplication
    info += "\nObject Properties (Relationships):\n"
    processed_props = set()  # To avoid repetition of properties
    if onto.object_properties():  # Check if there are any object properties
        for prop in onto.object_properties():
            if prop not in processed_props:  # Avoid duplicate property entries
                info += f"- {prop.name}:\n"
                if prop.comment:
                    info += f"  - Description: {prop.comment[0]}\n"
                # Avoid duplicate domains and ranges
                unique_domains = set(prop.domain)
                unique_ranges = set(prop.range)
                for domain in unique_domains:
                    info += f"  - Domain: {domain.name}\n"
                for range in unique_ranges:
                    info += f"  - Range: {range.name}\n"
                processed_props.add(prop)
    else:
        info += "  - No object properties defined in the ontology.\n"

    # Logical Statements (explicit relationships like CausesFailure) without duplication
    info += "\nLogical Statements (Relationships between instances):\n"
    processed_statements = set()  # To avoid repetition of statements
    statements_found = False  # Flag to track if any statements are found
    for prop in onto.object_properties():
        for s, o in prop.get_relations():
            statement = f"{s.name}.{prop.name}({o.name})"
            if statement not in processed_statements:  # Avoid duplicates
                info += statement + "\n"
                processed_statements.add(statement)
                statements_found = True
    if not statements_found:
        info += "  - No logical statements (relationships between instances) defined in the ontology.\n"
    return info

def verbalizer(logical_statement):
    """
    Converts a logical statement like 'piston_1.CausesFailure(oil_engine_1)' 
    or 'not piston_1.CausesFailure(electric_engine_1)' 
    to a natural language statement like 'Piston causes failure of oil engine'
    or 'Piston does not cause failure of electric engine'.
    """
    # Check for negation and strip 'not ' from the statement if present
    if logical_statement.startswith("not "):
        negation = True
        logical_statement = logical_statement[4:]  # Remove 'not '
    else:
        negation = False
    
    # Split the logical statement
    subject, property_with_object = logical_statement.split('.')
    property_, object_ = property_with_object.split('(')
    object_ = object_.strip(')')
    
    # Remove numbers and underscores from the subject and object
    subject_nl = subject.split('_')[0].capitalize()
    object_nl = object_.split('_')[0].capitalize()
    
    # Handle the negation case
    if negation:
        return f"{subject_nl} does not cause failure of {object_nl}"
    else:
        return f"{subject_nl} causes failure of {object_nl}"


import re
from owlready2 import *

def get_all_elements(onto):
    """
    Extracts all classes, properties, and individuals from the ontology.
    """
    classes = list(onto.classes())
    properties = list(onto.object_properties())  # Only object properties
    individuals = list(onto.individuals())

    return {
        "classes": classes,
        "properties": properties,
        "individuals": individuals
    }

def evaluate_statement_old(onto, subject_name, property_name, object_name):
    """
    Evaluates if a specific logical statement subject.property(object) is true or false in the ontology.
    """
    subject = onto.search_one(iri="*#" + subject_name)
    object_ = onto.search_one(iri="*#" + object_name)
    property_ = onto.search_one(iri="*#" + property_name)

    # Ensure that the individuals and property exist
    if not subject or not object_ or not property_:
        return False

    # Check if the property is an object property
    if not isinstance(property_, ObjectPropertyClass):
        return False

    # Check if the logical statement is true (if the object exists in the property of the subject)
    return object_ in property_[subject]

def evaluate_statement(onto, subject_name, property_name, object_name):
    """
    Evaluates if a specific logical statement subject.property(object) is true or false in the ontology.
    """
    subject = onto.search_one(iri="*#" + subject_name)
    object_ = onto.search_one(iri="*#" + object_name)
    property_ = onto.search_one(iri="*#" + property_name)

    # Ensure that the individuals and property exist
    if not subject or not object_ or not property_:
        return False

    # Check if the property is an object property
    if not isinstance(property_, ObjectPropertyClass):
        return False

    # Check if the logical statement is true (if the object exists in the property of the subject)
    return object_ in property_[subject]

def generate_and_evaluate_statements_old(onto):
    """
    Generates all possible logical statements and evaluates if they are true or false.
    """
    true_statements = []
    false_statements = []
    elements = get_all_elements(onto)

    # Iterate over each object property
    for prop in elements['properties']:
        # Get the domains and ranges of the property
        domains = prop.domain
        ranges = prop.range

        # Iterate over all individuals as possible subjects and objects
        for subject in elements['individuals']:
            for object_ in elements['individuals']:
                # Check if the individual's type matches the domain and range
                if any(isinstance(subject, domain) for domain in domains) and any(isinstance(object_, range_) for range_ in ranges):
                    # Create a logical statement
                    statement = f"{subject.name}.{prop.name}({object_.name})"
                    
                    # Evaluate the statement
                    is_true = evaluate_statement(onto, subject.name, prop.name, object_.name)
                    
                    # Add the statement to the appropriate list
                    if is_true:
                        true_statements.append(statement)
                    else:
                        false_statements.append(statement)

    return true_statements, false_statements



#########################
def generate_and_evaluate_statements(onto):
    """
    Generates all possible logical statements and evaluates if they are true or false.
    """
    true_statements = []
    false_statements = []
    all_statements = generate_all_combinations(onto)

    for statement in all_statements:
        # Extract subject, property, and object from the statement
        subject, property_with_object = statement.split('.')
        property_, object_ = property_with_object.split('(')
        object_ = object_.strip(')')

        # Evaluate if the statement is true or false
        is_true = evaluate_statement(onto, subject, property_, object_)

        # Add to the appropriate list
        if is_true:
            true_statements.append(statement)
        else:
            false_statements.append(statement)

    return true_statements, false_statements

def generate_all_combinations(onto):
    """
    Generates all possible logical statements without considering domain and range restrictions.
    """
    statements = []
    elements = get_all_elements(onto)

    # Iterate over each object property
    for prop in elements['properties']:
        # Iterate over all individuals as possible subjects and objects
        for subject in elements['individuals']:
            for object_ in elements['individuals']:
                # Create a logical statement
                statement = f"{subject.name}.{prop.name}({object_.name})"
                # Add to the list of statements (true or false not evaluated here)
                statements.append(statement)

    return statements

def verbalize_individual(individual_name, onto):
    """
    Extracts and verbalizes the name of an individual from the ontology.
    """
    # Find the individual in the ontology by name
    individual = onto.search_one(iri="*" + individual_name)
    if individual:
        # Use the individual name for verbalization
        return individual.name.split('_')[0].capitalize()
    else:
        return individual_name.capitalize()

def verbalize_property(property_name, onto):
    """
    Extracts and verbalizes the name of a property from the ontology.
    """
    # Find the property in the ontology by name
    property_ = onto.search_one(iri="*" + property_name)
    if property_:
        # Use the property name for verbalization
        return property_.name.split('_')[0].replace('CausesFailure', 'causes failure of').lower()
    else:
        return property_name.lower()


def generate_natural_language_variations(logical_statement, onto, is_negation=False):
    """
    Generates multiple natural language variations for a given logical statement, using the ontology to extract
    subject, object, and property information.
    """
    # Strip 'not ' for generating variations if the statement is negated
    if is_negation:
        logical_statement = logical_statement[4:]  # Remove 'not '

    # Extract subject, property, and object
    subject, property_with_object = logical_statement.split('.')
    property_, object_ = property_with_object.split('(')
    object_ = object_.strip(')')

    # Get the verbalized names of the subject, property, and object from the ontology
    subject_nl = verbalize_individual(subject, onto)
    object_nl = verbalize_individual(object_, onto)
    property_nl = verbalize_property(property_, onto)

    # Variations for positive and negative cases
    if is_negation:
        return [
            f"{subject_nl} does not {property_nl} {object_nl}",
            f"{subject_nl} is not responsible for {property_nl} {object_nl}",
            f"{property_nl.capitalize()} {object_nl} is not caused by {subject_nl}",
            f"The {property_nl} {object_nl} is not due to {subject_nl}",
        ]
    else:
        return [
            f"{subject_nl} {property_nl} {object_nl}",
            f"The {subject_nl.lower()} leads to {property_nl} {object_nl.lower()}.",
            f"{property_nl.capitalize()} {object_nl.lower()} is caused by the {subject_nl.lower()}.",
            f"The {object_nl.lower()}'s {property_nl} is due to the {subject_nl.lower()}.",
            f"{subject_nl.lower()} {property_nl} the {object_nl.lower()}",
        ]

def verbalizer(logical_statement, onto):
    """
    Converts a logical statement to a natural language statement using the ontology.
    """
    is_negation = logical_statement.startswith("not ")
    return generate_natural_language_variations(logical_statement, onto, is_negation=is_negation)[0]


def training_data_generator(onto):
    """
    Generates training data based on the ontology, including true and false statements
    and their corresponding natural language variations.
    """
    # Generate true and false logical statements test case
    #true_statements = ['battery_1.CausesFailure(electric_engine_1)']
    #false_statements = ['battery_1.CausesFailure(oil_engine_1)']
    
    # Generate true and false logical statements
    true_statements, false_statements = generate_and_evaluate_statements(onto)
    
    
    logical_statements = true_statements + false_statements
    
    # Create the training data with natural language and logical statement pairs
    data = []
    for logical_statement in logical_statements:
        # Generate variations for true statements
        if logical_statement in true_statements:
            natural_language_variations = generate_natural_language_variations(logical_statement, onto)
            for nl_variation in natural_language_variations:
                data.append((nl_variation, logical_statement))
        # Generate variations for false statements (both original and negated)
        else:
            natural_language_variations = generate_natural_language_variations(logical_statement, onto)
            for nl_variation in natural_language_variations:
                data.append((nl_variation, logical_statement))
            
            logical_statement_with_negation = f"not {logical_statement}"
            natural_language_variations_negated = generate_natural_language_variations(logical_statement_with_negation, onto, is_negation=True)
            for nl_variation in natural_language_variations_negated:
                data.append((nl_variation, logical_statement_with_negation))
    # Example output (training data):
    '''
    data = [
        ("Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)"),
        ("Battery does not cause failure of oil engine", "not battery_1.CausesFailure(oil_engine_1)"),
        ("Motor causes failure of electric engine", "motor_1.CausesFailure(electric_engine_1)"),
        ("Oil pump does not cause failure of electric engine", "not oil_pump_1.CausesFailure(electric_engine_1)")
    ]
    '''

    return data
############################
# Generate training data
#Test
if default:
    logical_statement_test='battery_1.CausesFailure(oil_engine_1)'
    natural_language_variations_test = generate_natural_language_variations(logical_statement_test, onto)
    print("logical_statement_test",logical_statement_test)
    print("natural_language_variations_test",natural_language_variations_test)
    logical_statement_with_negation_test = f"not {logical_statement_test}"
    natural_language_variations_test_negative = generate_natural_language_variations(logical_statement_with_negation_test, onto, is_negation=True)
    print("natural_language_variations_test_negative",natural_language_variations_test_negative)
elements = get_all_elements(onto)
print("elements", elements)
# Generate true and false logical statements
true_statements, false_statements = generate_and_evaluate_statements(onto)
logical_statements = true_statements + false_statements
print("logical_statements len",len(logical_statements))
training_data = training_data_generator(onto)
print("Training Data len",len(training_data))

# Run the reasoner to infer new knowledge
with onto:
    sync_reasoner()

# Enhanced function to check if a statement is true or false and provide reasons
def check_statement_with_details(onto, logical_form):
    """
    Evaluates if a logical statement is true or false within the ontology and provides detailed feedback.
    The logical statement should be in the format: `subject.property(object)` or `not subject.property(object)`.
    """
    # Parse the logical form using regular expressions
    match = re.match(r"(not )?(\w+)\.(\w+)\((\w+)\)", logical_form)
    if match:
        negation, subject_name, property_name, object_name = match.groups()

        # Search for the corresponding individuals and property in the ontology
        subject = onto.search_one(iri="*#" + subject_name)
        object_ = onto.search_one(iri="*#" + object_name)
        property_ = onto.search_one(iri="*#" + property_name)

        # Check if the parsed elements exist in the ontology
        if not subject or not object_ or not property_:
            return False, "The logical form references elements that do not exist in the ontology."

        # Check if the property is an object property
        if not isinstance(property_, ObjectPropertyClass):
            return False, f"{property_name} is not an object property."

        # Evaluate the logical statement
        is_true = object_ in property_[subject]

        if negation:
            if is_true:
                return False, f"{subject_name} actually has the property {property_name} with {object_name}, contradicting the statement."
            else:
                return True, None
        else:
            if is_true:
                return True, None
            else:
                # Collect possible reasons for the failure
                reasons = []

                # General reason based on class mismatches
                subject_classes = subject.is_a
                object_classes = object_.is_a

                # Check if the subject's class is disjoint with the object's class
                for subj_class in subject_classes:
                    for obj_class in object_classes:
                        if hasattr(obj_class, 'disjoint_with') and subj_class in obj_class.disjoint_with:
                            reasons.append(f"{subject_name} of class {subj_class.name} is disjoint with {object_name} of class {obj_class.name}.")

                # Additional explanation: class incompatibility
                elements = get_all_elements(onto)
                subject_class_names = [cls.name for cls in subject.is_a]
                object_class_names = [cls.name for cls in object_.is_a]

                # Cross-checking class types for incompatibility
                for subj_class in subject_class_names:
                    for obj_class in object_class_names:
                        if subj_class != obj_class:
                            reasons.append(f"{subject_name} of class {subj_class} cannot have a relationship with {object_name} of class {obj_class}.")

                return False, " ".join(reasons) if reasons else "No explicit reason found in the ontology."

    return False, "The logical form could not be parsed correctly."


# Function to evaluate a list of logical statements with details
def evaluate_statements_with_details(onto, statements):
    """
    Evaluates a list of logical statements in the given ontology and prints the results with details.
    """
    for stmt in statements:
        result, details = check_statement_with_details(onto, stmt)
        print(f"Statement: {stmt}")
        print(f"Result: {'True' if result else 'False'}")
        if details:
            print(f"Details: {details}")
        print("-" * 40)


# Generate training data
training_data = training_data_generator(onto)

from owlready2 import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# Function to generate the ML model
def model_generator(training_data):
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([x[0] for x in training_data])
    y_train = [x[1] for x in training_data]
    # Train a simple Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, vectorizer


# modelizer function to train and evaluate the model
def modelizer(onto):
    # Generate training data
    training_data = training_data_generator(onto)
    # Generate the model and vectorizer
    model, vectorizer = model_generator(training_data)
    return model, vectorizer
# Function to parse the input statement and generate the logical form
def convert_to_logical(statement, model, vectorizer):
    X_test = vectorizer.transform([statement])
    logical_form = model.predict(X_test)[0]
    return logical_form

# Function to generate evaluation results
def generate_evaluation_results(statement, logical_form, result, reason):
    evaluation_results = f"""
Evaluation Results:
Input statement: {statement}
Logical statement: {logical_form}
Statement: '{statement}' => Logical Form: '{logical_form}' => Result: {result}
Reason: {reason if reason else 'No contradiction found.'}
"""
    return evaluation_results

from owlready2 import *
def main(ontology_path):
    # Load your ontology
    onto = get_ontology(ontology_path).load()
    # Print out ontology information
    ontology_info = get_ontology_info(onto)
    print(ontology_info)
    # Run the reasoner to infer new knowledge
    with onto:
        sync_reasoner()
    # Generate training data
    #training_data = training_data_generator(onto)
    # Call main function to get the model and vectorizer
    model, vectorizer = modelizer(onto)
    return onto, model, vectorizer

# Function to handle evaluation and response generation with ontology
def handle_evaluation_with_ontology_old(user_input,statement, ontology_path=None):
    print("handle_evaluation_with_ontology")
    # Use the default ontology if no custom ontology is provided
    if ontology_path is None:
        ontology_path = "engine_ontology.owl"  # Default ontology path

    # Load the ontology using the provided or default path
    onto, model, vectorizer = main(ontology_path)
    logical_form = convert_to_logical(statement, model, vectorizer)
    
    result, reason = check_statement_with_details(onto, logical_form)
    evaluation_results = generate_evaluation_results(statement, logical_form, result, reason)

    if result:
        # True statements
        print("---- True Statement ----")
        print("evaluation_results:", evaluation_results)        
        # Extract ontology information dynamically
        ontology_info = get_ontology_info(onto)
        # Create dynamic LLM prompt
        llm_context = f"""
        User input:
        {user_input}
        Ontology input:
        {ontology_info}
        {evaluation_results}
        Statement:
        {statement}
        Logical statement:
        """
        prompt_template = """
        Given the ontology information and the evaluation results, 
        please provide the statements and logical statements that answer the user input based on the context.
        Context:
        {context}
        Statements:
        Logical statements:
        """
        # Fill the prompt template with the context
        prompt = PromptTemplate(input_variables=["context"], template=prompt_template)
        filled_prompt = prompt.format(context=llm_context)
        print("------------prompt to WatsonX for True Statement-----------",filled_prompt)
        # Generate a new correct statement using WatsonxLLM
        response = ask_watsonx(filled_prompt)
    else:
        # False statements
        print("---- False Statement ----")
        print("evaluation_results:", evaluation_results)
        
        # Extract ontology information dynamically
        ontology_info = get_ontology_info(onto)
        
        # Create dynamic LLM prompt
        llm_context = f"""
        User input:
        {user_input}
        Ontology input:
        {ontology_info}
        {evaluation_results}
        """
        
        prompt_template = """
        Given the ontology information and the evaluation results, 
        if the input statement provided is false, 
        please provide all correct statements and logical statements that answer the user input based on the context.
        Context:
        {context}
        Statements:
        Logical statements:
        """
        # Fill the prompt template with the context
        prompt = PromptTemplate(input_variables=["context"], template=prompt_template)
        filled_prompt = prompt.format(context=llm_context)
        print("------------prompt to WatsonX for False Statement-----------",filled_prompt)
        # Generate a new correct statement using WatsonxLLM
        response = ask_watsonx(filled_prompt)

    return response
import re


import re

def parse_response(response):
    """
    Extracts "Statements" and "Logical statements" from a response text, 
    handling variations in formatting and cleaning up the output.

    Args:
        response (str): The text containing the statements and logical statements.

    Returns:
        tuple: A tuple containing two lists:
            - extracted_statements: The cleaned statements section.
            - extracted_logical_statements: The cleaned logical statements section.
    """
    # Split the response into lines
    lines = response.split('\n')
    
    # Initialize variables to store the extracted sections
    extracted_statements = []
    extracted_logical_statements = []
    
    # Flags to check if we are in the desired section
    in_statements_section = False
    in_logical_statements_section = False
    
    # Iterate through each line to extract information
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        
        if "Statements:" in line and not in_logical_statements_section:
            in_statements_section = True
            in_logical_statements_section = False
            continue  # Skip the "Statements:" header
        
        if "Logical statements:" in line:
            in_statements_section = False
            in_logical_statements_section = True
            continue  # Skip the "Logical statements:" header
        
        # Collect data for statements section
        if in_statements_section and line and not line.startswith("Logical"):
            # To handle cases where multiple statements are on a single line
            statements = line.split("Statement:")
            for statement in statements:
                statement = statement.strip()
                if statement:
                    extracted_statements.append(statement)
        
        # Collect data for logical statements section
        if in_logical_statements_section and line:
            extracted_logical_statements.append(line)
    
    # Return the extracted statements and logical statements
    return extracted_statements, extracted_logical_statements
import json
import re

# Function to handle evaluation and response generation with ontology, returns structured JSON
def handle_evaluation_with_ontology_llama70b(user_input, statement, ontology_path=None):
    print("handle_evaluation_with_ontology")
    # Use the default ontology if no custom ontology is provided
    if ontology_path is None:
        ontology_path = "engine_ontology.owl"  # Default ontology path

    # Load the ontology using the provided or default path
    onto, model, vectorizer = main(ontology_path)
    logical_form = convert_to_logical(statement, model, vectorizer)
    
    result, reason = check_statement_with_details(onto, logical_form)
    evaluation_results = generate_evaluation_results(statement, logical_form, result, reason)

    if result:
        # True statements
        print("---- True Statement ----")
        print("evaluation_results:", evaluation_results)        
        # Extract ontology information dynamically
        ontology_info = get_ontology_info(onto)
        
        # Create dynamic LLM prompt
        llm_context = f"""
        User input:
        {user_input}
        Ontology input:
        {ontology_info}
        {evaluation_results}
        Statement:
        {statement}
        Logical statement:
        """
        
        prompt_template = """
        Given the ontology information and the evaluation results, 
        please provide the statements and the logical statements that answer the user input  based on the context.      
        
        Context:
        {context}
        Please complete the following that answer the user input based on the context.
        Statements:
        Logical statements:

        For example:
        Statements: subject property object
        Logical statements: subject.property(object)
        Explanation: Simple explanation

        """
        # Fill the prompt template with the context
        prompt = PromptTemplate(input_variables=["context"], template=prompt_template)
        filled_prompt = prompt.format(context=llm_context)
        print("------------prompt to WatsonX for True Statement-----------", filled_prompt)
        
        # Generate a new correct statement using WatsonxLLM
        response = ask_watsonx(filled_prompt)
        
        # Parse the response
        statements, logical_statements = parse_response(response)
        

        # Join the list of statements into a single string, separated by newlines
        statements_str = '\n'.join(statements).strip() if statements else ''
        logical_statements_str = '\n'.join(logical_statements).strip() if logical_statements else ''

        # Format the final JSON output (handle strings directly)
        output = {
            'statements': statements_str,
            'details': "Logical Statements (Relationships between instances): \n"+ logical_statements_str
        }


        return json.dumps(output)

    else:
        # False statements
        print("---- False Statement ----")
        print("evaluation_results:", evaluation_results)
        
        # Extract ontology information dynamically
        ontology_info = get_ontology_info(onto)
        
        # Create dynamic LLM prompt
        llm_context = f"""
        User input:
        {user_input}
        Ontology input:
        {ontology_info}
        {evaluation_results}
        """
        
        prompt_template = """
        Given the ontology information and the evaluation results, 
        if the input statement provided is false, 
        please provide the new correct statements and logical statements that answer the user input based on the context.
        Context:
        {context}

        Please complete the following that answer the user input based on the context.
        Statements:
        Logical statements:

        For example:
        Statements: subject property object
        Logical statements: subject.property(object)
        Explanation: Simple explanation

        """
        # Fill the prompt template with the context
        prompt = PromptTemplate(input_variables=["context"], template=prompt_template)
        filled_prompt = prompt.format(context=llm_context)
        print("------------prompt to WatsonX for False Statement-----------", filled_prompt)
        
        # Generate a new correct statement using WatsonxLLM
        response = ask_watsonx(filled_prompt)

        print("WatsonX response:", response)
        
        # Parse the response
        statements, logical_statements = parse_response(response)
        # Join the list of statements into a single string, separated by newlines
        statements_str = '\n'.join(statements).strip() if statements else ''
        logical_statements_str = '\n'.join(logical_statements).strip() if logical_statements else ''

        # Format the final JSON output (handle strings directly)
        output = {
            'statements': statements_str,
            'details': "Logical Statements (Relationships between instances): \n"+ logical_statements_str
        }

        return json.dumps(output)

# -----------------------------------------------------------------------
# Revised function using clearer prompts for True or False statements
# -----------------------------------------------------------------------
def handle_evaluation_with_ontology(user_input, statement, ontology_path=None):
    print("handle_evaluation_with_ontology")
    
    # Use default ontology if none provided
    if ontology_path is None:
        ontology_path = "engine_ontology.owl"  # Default path

    # Load the ontology + supporting components
    onto, model, vectorizer = main(ontology_path)
    
    # Convert user statement to a logical form
    logical_form = convert_to_logical(statement, model, vectorizer)
    
    # Evaluate statement truth
    result, reason = check_statement_with_details(onto, logical_form)
    
    # Summarize evaluation
    evaluation_results = generate_evaluation_results(statement, logical_form, result, reason)
    
    # Get textual ontology info
    ontology_info = get_ontology_info(onto)

    # Build context for the LLM
    llm_context = f"""
User input:
{user_input}

Ontology input:
{ontology_info}

{evaluation_results}
"""

    if result:
        # ------------------ True statement ------------------ #
        print("---- True Statement ----")
        print("evaluation_results:", evaluation_results)

        # More explicit instructions for True case
        prompt_template = """
You are a helpful AI assistant with knowledge of the provided ontology.

The input statement was determined to be TRUE.

{context}

Task:
Please provide statements and logical statements (and a short explanation) that directly answer the user input based on the given ontology context. 
Use the following format exactly (without additional commentary):

Statements:
<Your statements here, e.g. "Piston causes failure of oil engine.">

Logical statements:
<Your logical statements here, e.g. "piston_1.CausesFailure(oil_engine_1)">

Explanation:
<Short explanation here>
"""

        # Fill the prompt
        prompt = PromptTemplate(input_variables=["context"], template=prompt_template)
        filled_prompt = prompt.format(context=llm_context)
        print("------------ Prompt to LLM for True Statement -----------")
        print(filled_prompt)
        
        # Query your LLM
        response = ask_watsonx(filled_prompt)
        
        print("Response LLM",response)

        # Parse the LLM response
        statements, logical_statements = parse_response(response)

        # Prepare final JSON
        statements_str = '\n'.join(statements).strip() if statements else ''
        logical_statements_str = '\n'.join(logical_statements).strip() if logical_statements else ''

        output = {
            'statements': statements_str,
            'details': "Logical Statements (Relationships between instances): \n" + logical_statements_str
        }
        return json.dumps(output)

    else:
        # ------------------ False statement ------------------ #
        print("---- False Statement ----")
        print("evaluation_results:", evaluation_results)

        # More explicit instructions for False case
        prompt_template = """
You are a helpful AI assistant with knowledge of the provided ontology.

The input statement was determined to be FALSE.

{context}

Task:
Please provide the new correct statements and logical statements that answer the user input based on the ontology context. 
Use the following format exactly (without extra commentary).  
If there are multiple correct statements, you can list them all, but keep them short:

Statements:
<Your statements here, e.g. "Oil pump causes failure of oil engine.">

Logical statements:
<Your logical statements here, e.g. "oil_pump_1.CausesFailure(oil_engine_1)">

Explanation:
<Short explanation here>
"""

        # Fill the prompt
        prompt = PromptTemplate(input_variables=["context"], template=prompt_template)
        filled_prompt = prompt.format(context=llm_context)
        print("------------ Prompt to LLM for False Statement -----------")
        print(filled_prompt)
        
        # Query your LLM
        response = ask_watsonx(filled_prompt)
        print("LLM response:", response)
        
        # Parse the LLM response
        statements, logical_statements = parse_response(response)

        # Prepare final JSON
        statements_str = '\n'.join(statements).strip() if statements else ''
        logical_statements_str = '\n'.join(logical_statements).strip() if logical_statements else ''

        output = {
            'statements': statements_str,
            'details': "Logical Statements (Relationships between instances): \n" + logical_statements_str
        }
        return json.dumps(output)


# Function to handle evaluation and response generation without ontology
def handle_evaluation_without_ontology_clean(statement):
    print("handle_evaluation_without_ontology")
    
    prompt_template = """
    You are a helpful and friendly AI assistant. If you do not know the answer to a question, please be honest and say "I don't know" or "I don't have enough information to answer that."
    User Question:
    {statement}
    Assistant:
    """
    prompt = PromptTemplate(input_variables=["statement"], template=prompt_template)
    filled_prompt = prompt.format(statement=statement)
    print("filled_prompt", filled_prompt)
    response = ask_watsonx(filled_prompt)
    return response


def handle_evaluation_without_ontology_old(statement):
    print("handle_evaluation_without_ontology") 
    prompt_template = """
    You are a helpful and friendly AI assistant. 
    
    You will give a single statement of the causes of failures 
    based on the context that I will provide.  Just choose one statement.
    If you do not know the answer to a question, please be honest and say "I don't know" or "I don't have enough information to answer that."    
    I will provide context about causes of failures of a machines.
    context:
    Statement: Battery causes failure of oil engine
    Statement: Oil pump causes failure of electric engine
    Statement: Piston causes failure of electric engine
    User Question:
    {statement}
    Assistant:
    """
    prompt = PromptTemplate(input_variables=["statement"], template=prompt_template)
    filled_prompt = prompt.format(statement=statement)
    print("------------prompt to WatsonX without Ontology-----------", filled_prompt)
    response = ask_watsonx(filled_prompt)
    return response
def handle_evaluation_without_ontology_vecchio(statement):
    print("handle_evaluation_without_ontology") 
    prompt_template = """
    You are a helpful and friendly AI assistant. 
    
    You will give a single statement of the causes of failures 
    based on the context that I will provide. Just choose one statement.
    If you do not know the answer to a question, please be honest and say "I don't know" or "I don't have enough information to answer that."    
    I will provide context about causes of failures of a machines.
    context:
    Statement: Battery causes failure of oil engine
    Statement: Oil pump causes failure of electric engine
    Statement: Piston causes failure of electric engine
    User Question:
    {statement}
    Assistant:
    """
    prompt = prompt_template.format(statement=statement)
    print("------------prompt to WatsonX without Ontology-----------", prompt)
    
    # Assuming ask_watsonx returns a plain text response
    response = ask_watsonx(prompt)  # Example response: "The battery causes failure of the oil engine."
    
    # Ensure the response is properly formatted as JSON
    formatted_response = {
        "statements": response.strip(),  # Take the LLM's response as the statement
        "details": ""  # No details when ontology is not used
    }

    
    # Convert the formatted response to a JSON string
    return json.dumps(formatted_response)

def handle_evaluation_without_ontology(statement):
    print("handle_evaluation_without_ontology") 
    prompt_template = """
    You are a helpful and precise AI assistant. Your task is to determine the cause of failures based strictly on the provided context. 

    Instructions:
    - You must only answer based on the given context.
    - If a relevant statement exists in the context, respond with that exact statement.
    - If there is no matching information, respond with: "I don't know" or "I don't have enough information to answer that."

    Context:
    Statement: Battery causes failure of oil engine.
    Statement: Oil pump causes failure of electric engine.
    Statement: Piston causes failure of electric engine.
    Statement: Piston does not cause failure of oil engine.
    Statement: Oil pump does not cause failure of oil engine.
    Statement: Battery does not cause failure of electric engine.
    Statement: Motor does not cause failure of electric engine.

    User Question:
    {statement}

    Assistant:
    """
    prompt = prompt_template.format(statement=statement)
    print("------------prompt to WatsonX without Ontology-----------", prompt)
    
    # Assuming ask_watsonx returns a plain text response
    response = ask_watsonx(prompt)  # Example response: "The battery causes failure of the oil engine."
    
    # Ensure the response is properly formatted as JSON
    formatted_response = {
        "statements": response.strip(),  # Take the LLM's response as the statement
        "details": ""  # No details when ontology is not used
    }

    
    # Convert the formatted response to a JSON string
    return json.dumps(formatted_response)



# Unified function to handle evaluation based on ontology flag
def handle_evaluation_old(statement, use_ontology, ontology_path=None):
    if use_ontology:
        print("statement:", statement)
        answer_llm = handle_evaluation_without_ontology(statement)
     
        return handle_evaluation_with_ontology(statement,answer_llm, ontology_path)
    else:
        print("statement:", statement)
        return handle_evaluation_without_ontology(statement)
    
import json

# Unified function to handle evaluation based on ontology flag
def handle_evaluation(statement, use_ontology, ontology_path=None):
    if use_ontology:
        print("statement:", statement)
        # Get the response from handle_evaluation_without_ontology
        answer_llm_json = handle_evaluation_without_ontology(statement)
        # Parse the JSON string into a dictionary
        answer_llm_dict = json.loads(answer_llm_json)
        # Extract the 'statements' field
        answer_llm = answer_llm_dict['statements']
        # Call the function with ontology, passing the 'answer_llm'
        return handle_evaluation_with_ontology(statement, answer_llm, ontology_path)
    else:
        print("statement:", statement)
        # Directly return the response when ontology is not used
        return handle_evaluation_without_ontology(statement)
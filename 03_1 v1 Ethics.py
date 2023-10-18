risks_list = [
        "Privacy concerns: AI-powered devices may collect personal data, raising concerns about how this data is stored and used.",
        "Security vulnerabilities: AI systems could be susceptible to cyberattacks, leading to unauthorized access to personal information.",
        "Bias in AI algorithms: AI algorithms might inadvertently perpetuate biases present in the data they are trained on, leading to unfair or discriminatory outcomes.",
        #you may add more AI Risks to this list
    ]


def get_risks_and_categories(risks_list):
    """
    Identify and categorize potential risks of using AI in everyday life.
    Parameters:
        risks_list (list): A list of potential risks and categories associated with AI usage.
    Returns:
        list: A list containing categorized risks.
    """

    print("Welcome to the AI Risks Identification Tool!")
    # Initialize an empty list to store categorized risks
    risks_categories = []

    for risk in risks_list:
        # Split each element in the risks_list into lines
        lines = risk.split("\n")

        # Initialize variables to store category and risk description
        category = ""
        risk_description = ""

        for line in lines:
            if "Category:" in line:
                # Extract the category from the line
                category = line.split(":")[1].strip()
            else:
                # If the line doesn't contain "Category:", it's part of the risk description
                risk_description += line

        # Append the category and risk description as a list to the risks_categories list
        risks_categories.append([category, risk_description])

    # Return the list of categorized risks
    return risks_categories

# @test_identify_risks
def identify_risks(risks_categories):
    """
    Identify and categorize potential risks of using AI in everyday life.

    Parameters:
        risks_categories (list): A list of potential risks and categories associated with AI usage.

    Returns:
        dict: A dictionary containing categorized risks.
    """

    # Initialize an empty dictionary to store categorized risks
    categorized_risks = {}

    for category, risk_description in risks_categories:


    # If the category is not already a key in the dictionary, add it


        if category not in categorized_risks:
            categorized_risks[category] = []

            categorized_risks[category] = []

            categorized_risks
            # Append the risk description to the list under the corresponding category
        categorized_risks[category].append(risk_description)

        # categorized_risks[category].append(risk
        #
        # categorized_risks[category].append(r
        #
        # categorized_risks[category
        #
        # categorized_r
        #
        # categorized
    return categorized_risks


def display_risks_summary(risks):
    """
    Display a summary of identified risks under each category.

    Parameters:
        risks (dict): A dictionary containing categorized risks.
    """
    print("\nAI Risks Summary:\n")

    return None

risks_categories = get_risks_and_categories(risks_list)
print(f"risks_categories: {risks_categories}")
risks_data = identify_risks(risks_categories)
display_risks_summary(risks_data)

"""
Name: Efe ATEŞ
Student No: 22290588

"""

import os


#Market
def market_value_low(price):
    if price <= 0:
        return 0
    elif 0 < price <= 50:
        return price / 50  
    elif 50 < price < 100:
        return (100 - price) / 50  
    else:
        return 0

def market_value_medium(price):
    if price <= 50 or price >= 250:
        return 0
    elif 50 < price < 100:
        return (price - 50) / 50  
    elif 100 <= price <= 200:
        return 1  
    else:  
        return (250 - price) 

def market_value_high(price):
    if price <= 200 or price >= 700:
        return 0
    elif 200 < price < 300:
        return (price - 200) / 100  
    elif 300 <= price <= 600:
        return 1  
    else:  # 600 < price < 700
        return (700 - price) / 100  

def market_value_very_high(price):
    if price <= 600:
        return 0
    elif 600 < price < 700:
        return (price - 600) / 100  
    else: 
        return 1  

#Location 
def location_bad(loc):
    if loc <= 0:
        return 1  
    elif 0 < loc <= 2:
        return 1  
    elif 2 < loc < 3:
        return (3 - loc)  
    else:
        return 0

def location_fair(loc):
    if loc <= 2 or loc >= 8:
        return 0
    elif 2 < loc < 5:
        return (loc - 2) / 3  
    else:  # 5 < loc < 8
        return (8 - loc) / 3  
def location_excellent(loc):
    if loc <= 7:
        return 0
    elif 7 < loc < 8:
        return (loc - 7)  
    else:  # loc >= 8
        return 1  


#Person's Asset

def asset_low(asset):
    if asset <= 0:
        return 1
    elif 0 < asset < 200:
        return (200 - asset) / 200
    else:
        return 0

def asset_medium(asset):
    if asset <= 100 or asset >= 600:
        return 0
    elif 100 < asset < 200:
        return (asset - 100) / 100
    elif 200 <= asset <= 500:
        return 1
    else:
        return (600 - asset) / 100

def asset_high(asset):
    if asset <= 500:
        return 0
    elif 500 < asset < 600:
        return (asset - 500) / 100
    else:
        return 1
#Person's Income
def income_low(income):
    if income <= 10:
        return 1
    elif 10 < income < 20:
        return (20 - income) / 10
    else:
        return 0

def income_medium(income):
    if income <= 20 or income >= 50:
        return 0
    elif 20 < income < 30:
        return (income - 20) / 10
    elif income == 30:
        return 1
    else:
        return (50 - income) / 20

def income_high(income):
    if income <= 40 or income >= 80:
        return 0
    elif 40 < income < 60:
        return (income - 40) / 20
    elif income == 60:
        return 1
    else:
        return (80 - income) / 20

def income_very_high(income):
    if income <= 70:
        return 0
    elif 70 < income < 80:
        return (income - 70) / 10
    else:
        return 1
#Interest
def interest_low(interest):
    if interest <= 2:
        return 1
    elif 2 < interest < 4:
        return (4 - interest) / 2
    else:
        return 0

def interest_medium(interest):
    if interest <= 2 or interest >= 8:
        return 0
    elif 2 < interest < 4:
        return (interest - 2) / 2
    elif 4 <= interest <= 6:
        return 1
    else:
        return (8 - interest) / 2

def interest_high(interest):
    if interest <= 6:
        return 0
    elif 6 < interest < 8:
        return (interest - 6) / 2
    else:
        return 1


def houseEval(price,loc):

    very_low_activation = 0
    low_activation = 0
    medium_activation = 0
    high_activation = 0
    very_high_activation = 0
    
    rule1 = market_value_low(price)
    low_activation = max(low_activation, rule1)
    
    rule2 = location_bad(loc)
    low_activation = max(low_activation, rule2)
    
    rule3 = min(location_bad(loc), market_value_low(price))
    very_low_activation = max(very_low_activation, rule3)
    
    rule4 = min(location_bad(loc), market_value_medium(price))
    low_activation = max(low_activation, rule4)
    
    rule5 = min(location_bad(loc), market_value_high(price))
    medium_activation = max(medium_activation, rule5)
    
    rule6 = min(location_bad(loc), market_value_very_high(price))
    high_activation = max(high_activation, rule6)
    
    rule7 = min(location_fair(loc), market_value_low(price))
    low_activation = max(low_activation, rule7)
    
    rule8 = min(location_fair(loc), market_value_medium(price))
    medium_activation = max(medium_activation, rule8)
    
    rule9 = min(location_fair(loc), market_value_high(price))
    high_activation = max(high_activation, rule9)
    
    rule10 = min(location_fair(loc), market_value_very_high(price))
    very_high_activation = max(very_high_activation, rule10)
    
    rule11 = min(location_excellent(loc), market_value_low(price))
    medium_activation = max(medium_activation, rule11)
    
    rule12 = min(location_excellent(loc), market_value_medium(price))
    high_activation = max(high_activation, rule12)
    
    rule13 = min(location_excellent(loc), market_value_high(price))
    very_high_activation = max(very_high_activation, rule13)
    
    rule14 = min(location_excellent(loc), market_value_very_high(price))
    very_high_activation = max(very_high_activation, rule14)
    
    house_evaluations = {
        "Very_low": very_low_activation,
        "Low": low_activation,
        "Medium": medium_activation,
        "High": high_activation,
        "Very_high": very_high_activation
    }
    
    return house_evaluations

def applicantEval(asset, income):
    very_low_activation = 0
    low_activation = 0
    medium_activation = 0
    high_activation = 0
    very_high_activation= 0

    #If (Asset is Low) and (Income is Low) then (Applicant is Low)
    rule1 = min(asset_low(asset), income_low(income))
    low_activation = max(low_activation, rule1)
    
    #If (Asset is Low) and (Income is Medium) then (Applicant is Low)
    rule2 = min(asset_low(asset), income_medium(income))
    low_activation = max(low_activation, rule2)
    
    #If (Asset is Low) and (Income is High) then (Applicant is Medium)
    rule3 = min(asset_low(asset), income_high(income))
    medium_activation = max(medium_activation, rule3)
    
    #If (Asset is Low) and (Income is Very_high) then (Applicant is High)
    rule4 = min(asset_low(asset), income_very_high(income))
    high_activation = max(high_activation, rule4)
    
    #If (Asset is Medium) and (Income is Low) then (Applicant is Low)
    rule5 = min(asset_medium(asset), income_low(income))
    low_activation = max(low_activation, rule5)
    
    #If (Asset is Medium) and (Income is Medium) then (Applicant is Medium)
    rule6 = min(asset_medium(asset), income_medium(income))
    medium_activation = max(medium_activation, rule6)
    
    #If (Asset is Medium) and (Income is High) then (Applicant is High)
    rule7 = min(asset_medium(asset), income_high(income))
    high_activation = max(high_activation, rule7)
    
    #If (Asset is Medium) and (Income is Very_high) then (Applicant is High)
    rule8 = min(asset_medium(asset), income_very_high(income))
    high_activation = max(high_activation, rule8)
    
    #If (Asset is High) and (Income is Low) then (Applicant is Medium)
    rule9 = min(asset_high(asset), income_low(income))
    medium_activation = max(medium_activation, rule9)
    
    #If (Asset is High) and (Income is Medium) then (Applicant is Medium)
    rule10 = min(asset_high(asset), income_medium(income))
    medium_activation = max(medium_activation, rule10)
    
    #If (Asset is High) and (Income is High) then (Applicant is High)
    rule11 = min(asset_high(asset), income_high(income))
    high_activation = max(high_activation, rule11)
    
    #If (Asset is High) and (Income is Very_high) then (Applicant is High)
    rule12 = min(asset_high(asset), income_very_high(income))
    high_activation = max(high_activation, rule12)
    
    applicant_evaluations = {
        "Low": low_activation,
        "Medium": medium_activation,
        "High": high_activation
    }
    
    return applicant_evaluations


def amountCredit(house_eval, applicant_eval, income, interest):
    very_low_activation= 0
    low_activation = 0
    medium_activation = 0
    high_activation = 0
    very_high_activation= 0

    rule1 = min(income_low(income),interest_medium(interest))
    very_low_activation = max(very_low_activation,rule1)

    rule2 = min(income_low(income),interest_high(interest))
    very_low_activation = max(very_low_activation,rule2)

    rule3 = min(income_medium(income), interest_high(interest))
    low_activation = max(low_activation, rule3)
    
    #dictionary search
    rule4 = applicant_eval["Low"]
    very_low_activation = max(very_low_activation, rule4)
    
    rule5 = house_eval["Very_low"]
    very_low_activation = max(very_low_activation, rule5)
    
    rule6 = min(applicant_eval["Medium"], house_eval["Very_low"])
    low_activation = max(low_activation, rule6)
    
    rule7 = min(applicant_eval["Medium"], house_eval["Low"])
    low_activation = max(low_activation, rule7)
    
    rule8 = min(applicant_eval["Medium"], house_eval["Medium"])
    medium_activation = max(medium_activation, rule8)
    
    rule9 = min(applicant_eval["Medium"], house_eval["High"])
    high_activation = max(high_activation, rule9)
    
    rule10 = min(applicant_eval["Medium"], house_eval["Very_high"])
    high_activation = max(high_activation, rule10)
    
    rule11 = min(applicant_eval["High"], house_eval["Very_low"])
    low_activation = max(low_activation, rule11)
    
    rule12 = min(applicant_eval["High"], house_eval["Low"])
    medium_activation = max(medium_activation, rule12)
    
    rule13 = min(applicant_eval["High"], house_eval["Medium"])
    high_activation = max(high_activation, rule13)
    
    rule14 = min(applicant_eval["High"], house_eval["High"])
    high_activation = max(high_activation, rule14)
    
    rule15 = min(applicant_eval["High"], house_eval["Very_high"])
    very_high_activation = max(very_high_activation, rule15)
    
    credit_evaluations = {
        "Very_low": very_low_activation,
        "Low": low_activation,
        "Medium": medium_activation,
        "High": high_activation,
        "Very_high": very_high_activation
    }
    
    return credit_evaluations

    

def evaluateThePerson(market_value, location, asset, income, interest):
    HouseResult = houseEval(market_value, location)
    appResult = applicantEval(asset, income)
    creditResult =  amountCredit(HouseResult, appResult, income, interest)


    #defuzzyfication part 2 crisp
    credit_values = {"Very_low": 0, "Low": 25, "Medium": 50, "High": 75, "Very_high": 100}
    numerator = 0
    denominator = 0

    for category, activation in creditResult.items():
        numerator += credit_values[category] * activation
        denominator += activation

    if denominator == 0:
        crisp_result = 0
    else:
        crisp_result = numerator / denominator

    return {
        "House Evaluation": HouseResult,
        "Applicant Evaluation": appResult,
        "Credit Evaluation": creditResult,
        "Crisp Credit Score": crisp_result
    }





            

if __name__ == "__main__":
    person1 = evaluateThePerson(market_value=600, location=4, asset=1000, income=890, interest=3)
    print("-------------------------------\nPERSON 1",end="\n-------------------------------\n")
    print("House Evaluation \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t Applicant Score \t\t\t\t\t\t\t\t\t\t\t\t Credit Score\t\t\t\t\t\t\t Crisp Credit Score")    
    print(str(person1['House Evaluation']).strip('{}'),str(person1['Applicant Evaluation']).strip('{}'),str(person1['Credit Evaluation']).strip('{}'),(person1['Crisp Credit Score']),sep=" |*| ")

    person2 = evaluateThePerson(market_value=900, location=8, asset=10000, income=8900, interest=1)
    print("-------------------------------\nPERSON 2",end="\n-------------------------------\n")
    print("House Evaluation \t\t\t\t\t\t\t\t\t\t\t\t\t Applicant Score \t\t\t\t\t\t\t\t\t\t\t\t Credit Score\t\t\t\t\t\t\t Crisp Credit Score")    
    print(str(person2['House Evaluation']).strip('{}'),str(person2['Applicant Evaluation']).strip('{}'),str(person2['Credit Evaluation']).strip('{}'),(person2['Crisp Credit Score']),sep=" |*| ")


    
    person3 = evaluateThePerson(market_value=200, location=1, asset=10000, income=900, interest=10)
    print("-------------------------------\nPERSON 3",end="\n-------------------------------\n")
    print("House Evaluation \t\t\t\t\t\t\t\t\t\t\t\t\t Applicant Score \t\t\t\t\t\t\t\t\t\t\t\t Credit Score\t\t\t\t\t\t\t Crisp Credit Score")
    print(str(person3['House Evaluation']).strip('{}'),str(person3['Applicant Evaluation']).strip('{}'),str(person3['Credit Evaluation']).strip('{}'),(person3['Crisp Credit Score']),sep=" |*| ")

    
    
    
print()
os.system("pause")



    
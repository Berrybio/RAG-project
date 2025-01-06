#!/usr/bin/env python
# coding: utf-8

import requests
import pandas as pd

# Initial URL for the first API call
base_url = "https://clinicaltrials.gov/api/v2/studies"
params = {
    # "query.titles": "Diabetes", #need change# 
    "pageSize": 100,
    "query.intr": "DRUG",
    "query.cond": "Alzheimer", 
    "query.term": "Phase2", 
}

# Initialize an empty list to store the data
data_list = []

# Loop until there is no nextPageToken
while True:
    # Print the current URL (for debugging purposes)
    print("Fetching data from:", base_url + '?' + '&'.join([f"{k}={v}" for k, v in params.items()]))
    
    # Send a GET request to the API
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        studies = data.get('studies', [])  # Extract the list of studies

        # Loop through each study and extract specific information
        # need extract the inclusion and exclusion criteria # 
        for study in studies:
            # Safely access nested keys
            nctId = study['protocolSection']['identificationModule'].get('nctId', 'Unknown')
            overallStatus = study['protocolSection']['statusModule'].get('overallStatus', 'Unknown')
            startDate = study['protocolSection']['statusModule'].get('startDateStruct', {}).get('date', 'Unknown Date')
            conditions = ', '.join(study['protocolSection']['conditionsModule'].get('conditions', ['No conditions listed']))
            acronym = study['protocolSection']['identificationModule'].get('acronym', 'Unknown')

            # Extract interventions safely
            interventions_list = study['protocolSection'].get('armsInterventionsModule', {}).get('interventions', [])
            interventions = ', '.join([intervention.get('name', 'No intervention name listed') for intervention in interventions_list]) if interventions_list else "No interventions listed"
            
            # Extract locations safely
            locations_list = study['protocolSection'].get('contactsLocationsModule', {}).get('locations', [])
            locations = ', '.join([f"{location.get('city', 'No City')} - {location.get('country', 'No Country')}" for location in locations_list]) if locations_list else "No locations listed"
            
            # Extract dates and phases
            primaryCompletionDate = study['protocolSection']['statusModule'].get('primaryCompletionDateStruct', {}).get('date', 'Unknown Date')
            studyFirstPostDate = study['protocolSection']['statusModule'].get('studyFirstPostDateStruct', {}).get('date', 'Unknown Date')
            lastUpdatePostDate = study['protocolSection']['statusModule'].get('lastUpdatePostDateStruct', {}).get('date', 'Unknown Date')
            studyType = study['protocolSection']['designModule'].get('studyType', 'Unknown')
            phases = ', '.join(study['protocolSection']['designModule'].get('phases', ['Not Available']))

            # Append the data to the list as a dictionary
            data_list.append({
                "NCT ID": nctId,
                "Acronym": acronym,
                "Overall Status": overallStatus,
                "Start Date": startDate,
                "Conditions": conditions,
                "Interventions": interventions,
                "Locations": locations,
                "Primary Completion Date": primaryCompletionDate,
                "Study First Post Date": studyFirstPostDate,
                "Last Update Post Date": lastUpdatePostDate,
                "Study Type": studyType,
                "Phases": phases
            })

        # Check for nextPageToken and update the params or break the loop
        nextPageToken = data.get('nextPageToken')
        if nextPageToken:
            params['pageToken'] = nextPageToken  # Set the pageToken for the next request
        else:
            break  # Exit the loop if no nextPageToken is present
    else:
        print("Failed to fetch data. Status code:", response.status_code)
        break

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)

# Print the DataFrame
print(df)

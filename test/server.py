import requests
import pandas as pd

# List of all debriefs
debriefs = [
    {
        "id": "DB561",
        "background": """Overview: In this operation, initial weather forecasts were clear, and all communication systems were confirmed as fully operational before departure. Key Issues: Mid-route, an unexpected, intense storm disrupted both the digital communication channels and the automated navigation system, causing significant ambiguity in the received signals. Resolution: Drivers immediately resorted to manual override protocols, using backup comm channels and pre-verified alternative routes to continue safely under the adverse weather conditions.""",
        "trip_progress": """Timeline: 08:00 AM - Depart with clear weather and stable comms; 08:50 AM - Unexpected heavy storm triggers system disruptions; 09:15 AM - Manual override engaged; 09:45 AM - Backup communication channels re-established; 10:30 AM - Transit resumes safely; 10:50 AM - Continued progression under manual navigation.""",
        "route_consideration": """Route Consideration: The chosen route was re-assessed based on updated weather data. Planners selected alternative pathways with natural shelter and historically lower wind exposure. These routes provided a dependable setting for manual control, thereby mitigating risks of further communication loss and ensuring a safer, albeit slower, transit."""
    },
    {
        "id": "DB562",
        "background": """Overview: Pre-trip schedules were extremely tight and did not allow enough downtime for driver recovery. Key Issues: During transit, the lack of sufficient rest led to pronounced driver fatigue, which resulted in a critical safety checkpoint being missed, jeopardizing the mission's overall schedule. Resolution: Emergency stops and enforced rest breaks were promptly administered, and the checkpoint was successfully re-attempted with improved alertness.""",
        "trip_progress": """Timeline: 07:45 AM - Depart with signs of fatigue; 08:40 AM - Critical checkpoint missed; 09:00 AM - Emergency stop initiated for rest; 09:30 AM - Additional rest period provided; 10:10 AM - Checkpoint successfully reattempted; 10:40 AM - Transit resumes; 11:00 AM - Schedule recovered.""",
        "route_consideration": """Route Consideration: The route was restructured to include shorter segments with additional checkpoints and designated rest stops along the way. This minimized the continuous drive duration and allowed for immediate driver recovery. The new routing plan also offered alternate paths in the event of further delays."""
    },
    {
        "id": "DB563",
        "background": """Overview: Thorough pre-trip cargo inspections were conducted across the fleet, which revealed several cases of load imbalance that could potentially compromise vehicle stability. Key Issues: These imbalances were immediately flagged as critical because uneven loads can affect braking and handling. Resolution: Instant corrective measures were taken to rebalance the loads, and continuous, periodic in-transit load verifications confirmed that the cargo remained securely and evenly distributed throughout the journey.""",
        "trip_progress": """Timeline: 07:50 AM - Depart after corrective load adjustments; 08:30 AM - Load verification checkpoint conducted; 08:50 AM - Cargo rebalancing confirmed; 09:20 AM - Further in-transit checks validate load stability; 09:50 AM - Transit proceeds securely; 10:15 AM - Final verification completed.""",
        "route_consideration": """Route Consideration: The route was chosen based on its even and stable road surfaces, which naturally reduce the risk of cargo shift due to abrupt maneuvers. A detailed road condition analysis was performed to ensure that the selected path would support consistent load stability, reducing risks during sudden stops or accelerations."""
    },
    {
        "id": "DB564",
        "background": """Overview: All on-board systems were thoroughly tested before departure, and pre-trip diagnostics confirmed that the navigation modules were functioning correctly. Key Issues: However, mid-transit, unexpected software glitches resulted in several navigation errors which caused deviation from the planned route and increased risk to safety. Resolution: Drivers quickly engaged manual override protocols, relying on a pre-established backup route and supplementary GPS data to restore the correct course.""",
        "trip_progress": """Timeline: 08:05 AM - Depart with normal navigation settings; 08:55 AM - Navigation error detected due to software glitch; 09:15 AM - Manual override is activated; 09:50 AM - Backup route engaged; 10:30 AM - Transit stabilized under manual control; 10:50 AM - Navigation returns to normal.""",
        "route_consideration": """Route Consideration: Following the error, the route was re-assessed and reconfigured to steer clear of areas known for similar technical issues. Planners used historical data and alternative navigation pathways to select a route with minimal interference, ensuring a safer passage for the manually controlled section of the journey."""
    },
    {
        "id": "DB565",
        "background": """Overview: A full preventive maintenance cycle was completed before departure, ensuring that all systems were fully operational and up to date. Key Issues: During transit, minor diagnostic alerts were triggered by routine system fluctuations, yet these were anticipated given the comprehensive maintenance effort. Resolution: Scheduled diagnostic stops allowed maintenance crews to address and resolve the alerts quickly, thereby preventing any disruption to the overall operation.""",
        "trip_progress": """Timeline: 08:00 AM - Depart after thorough maintenance clearance; 09:00 AM - Scheduled diagnostic stop conducted; 09:45 AM - Minor alerts resolved successfully; 10:30 AM - Transit resumes smoothly; 11:15 AM - Operation concludes with all systems stable; 11:30 AM - Final review confirms efficiency.""",
        "route_consideration": """Route Consideration: The route was selected after a detailed review of service center locations along the path, ensuring easy access to technical support whenever necessary. Road quality, traffic patterns, and accessibility to maintenance facilities were all considered to guarantee a reliable route that would not compromise vehicle performance."""
    }
]

# Combine mandatory fields into a test payload
test_payload = {
    "texts": [
        f"{d['background']} {d['trip_progress']} {d['route_consideration']}"
        for d in debriefs
    ]
}

# Convert to a DataFrame for visual checking or saving
test_df = pd.DataFrame(test_payload["texts"], columns=["Combined Text"])
test_df["Debrief ID"] = [d["id"] for d in debriefs]

# Reorder columns
test_df = test_df[["Debrief ID", "Combined Text"]]

print(test_df.head())

url = "http://127.0.0.1:4005/predict"

# Send POST request
response = requests.post(url, json={"texts": test_df["Combined Text"].tolist()})

# Process response
if response.status_code == 200:
    predictions = response.json()
    for i, item in enumerate(predictions):
        print(f"\nSample #{i + 1}")
        print("Text:           ", item['text'])
        print("Predicted Tags: ", ", ".join(item['predicted_tags']))
else:
    print("❌ Error:", response.status_code)
    print(response.text)

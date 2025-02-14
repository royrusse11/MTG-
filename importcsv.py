import csv
import requests
import time
import os  # Import the os module

def fetch_card_details(uuid):
    print(f"Attempting to fetch details for UUID: {uuid}")
    url = f"https://api.scryfall.com/cards/{uuid}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        rate_limit_remaining = response.headers.get('X-Ratelimit-Remaining')
        rate_limit_reset = response.headers.get('X-Ratelimit-Reset')
        if rate_limit_reset:
            reset_time = int(rate_limit_reset) - time.time()
            print(f"Rate limit will reset in {reset_time} seconds")
        print(f"Successfully fetched details for UUID: {uuid}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch details for UUID {uuid}: {e}")
        return None

def update_card_collection(file_path):
    file_path = os.path.expanduser(file_path)  # Expand the ~ to the full home directory path
    updated_cards = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        for row in reader:
            if len(row) < 2:
                print("Skipping row with insufficient columns")
                continue
            uuid = row[1]
            card_info = fetch_card_details(uuid)
            if card_info:
                useful_data = {
                    'scryfall_uuid': card_info.get('id'),
                    'name': card_info.get('name'),
                    'type_line': card_info.get('type_line'),
                    'cmc': card_info.get('cmc'),
                    'color_identity': ','.join(card_info.get('color_identity', [])),
                    'set': card_info.get('set'),
                    'rarity': card_info.get('rarity'),
                    'power': card_info.get('power'),
                    'toughness': card_info.get('toughness'),
                    'oracle_text': card_info.get('oracle_text', 'N/A')  # Fetch the description
                }
                updated_cards.append(useful_data)
            else:
                print(f"Skipping UUID {uuid} due to fetch error.")
            time.sleep(0.2)
    return updated_cards

def save_to_csv(cards, filename):
    filename = os.path.expanduser(filename)  # Expand the ~ to the full home directory path
    if not cards:
        print("No cards to save.")
        return
    fieldnames = ['scryfall_uuid', 'name', 'type_line', 'cmc', 'color_identity', 'set', 'rarity', 'power', 'toughness', 'oracle_text']
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(cards)
    print(f"Card details saved to {filename}")

# Example usage
cards_with_details = update_card_collection('~/Desktop/MTGProject/Card_Binder_export.csv')
if cards_with_details:
    save_to_csv(cards_with_details, '~/Desktop/MTGProject/Updated_Card_Details.csv')
else:
    print("No cards were fetched. Check the input file and API responses.")

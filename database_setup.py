import sqlite3

def create_database():
    conn = sqlite3.connect("whitelist.db")
    cursor = conn.cursor()

    # Nettoyer l'existant si nécessaire
    cursor.execute("DROP TABLE IF EXISTS authorized_vehicles")

    # Créer la table
    cursor.execute("""
    CREATE TABLE authorized_vehicles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT NOT NULL,
        normalized_plate TEXT NOT NULL,
        owner TEXT NOT NULL
    )
    """)

    # Données fournies par l'utilisateur
    # On stocke la version brute (avec tirets) et la version normalisée (sans espaces/tirets) pour la comparaison
    data = [
        ("42444-أ-1", "M.ACHARIFI"),
        ("23242-أ-55", "M.EL HASNI"),
        ("15555-ه-1", "M.AZIZ"),
        ("78904-ه-6", "M.HAFID")
    ]

    print("Insertion des données...")
    for plate, owner in data:
        # Normalisation : on enlève espaces et tirets pour faciliter la recherche
        normalized = plate.replace("-", "").replace(" ", "")
        cursor.execute("INSERT INTO authorized_vehicles (plate_number, normalized_plate, owner) VALUES (?, ?, ?)", 
                       (plate, normalized, owner))
        print(f"✅ Ajouté : {plate} ({owner})")

    conn.commit()
    conn.close()
    print("\nBase de données 'whitelist.db' créée avec succès !")

if __name__ == "__main__":
    create_database()

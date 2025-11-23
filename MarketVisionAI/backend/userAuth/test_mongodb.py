from pymongo import MongoClient
from datetime import datetime

def quick_test():
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Check database and collection
        db = client['auth-app']
        collections = db.list_collection_names()
        print(f"📊 Collections in auth-app: {collections}")
        
        # Check if users collection exists
        if 'users' in collections:
            users_collection = db['users']
            user_count = users_collection.count_documents({})
            print(f"👥 Users collection exists with {user_count} documents")
            
            # Show all users
            users = list(users_collection.find().limit(5))
            if users:
                print("\n📋 Current users:")
                for user in users:
                    print(f"  - {user.get('firstName')} {user.get('lastName')} ({user.get('email')})")
            else:
                print("ℹ️  No users found in database")
        else:
            print("❌ Users collection does not exist")
            
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
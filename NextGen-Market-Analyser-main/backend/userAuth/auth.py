from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import os
from datetime import datetime, timedelta
import jwt

app = Flask(__name__)
CORS(app)

# 🔐 JWT Secret Key (Use a secure key in production)
app.config['SECRET_KEY'] = "your_super_secret_key_here"

# MongoDB connection
try:
    MONGODB_URI = 'mongodb://localhost:27017/'
    client = MongoClient(MONGODB_URI)
    client.admin.command('ping')
    print("✅ Connected to MongoDB!")

    db = client['stockdb']
    users_collection = db['users']

except Exception as e:
    print(f"❌ MongoDB error: {e}")


# ✅ Health check route
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'success': True, 'message': 'Server is running'})


# ✅ Signup Route
@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()

        required_fields = ['firstName', 'lastName', 'email', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'message': f'{field} is required'}), 400

        email = data['email'].lower().strip()

        if users_collection.find_one({'email': email}):
            return jsonify({'success': False, 'message': 'User already exists'}), 400

        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

        user_data = {
            'firstName': data['firstName'].strip(),
            'lastName': data['lastName'].strip(),
            'email': email,
            'password': hashed_password,
            'phone': data.get('phone', '').strip(),
            'companyName': data.get('companyName', '').strip(),
            'country': data.get('country', '').strip(),
            'role': data.get('role', '').strip(),
            'createdAt': datetime.utcnow()
        }

        result = users_collection.insert_one(user_data)

        return jsonify({
            'success': True,
            'message': 'Signup successful',
            'user': {
                'id': str(result.inserted_id),
                'firstName': user_data['firstName'],
                'lastName': user_data['lastName'],
                'email': user_data['email']
            }
        }), 201

    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'success': False, 'message': 'Server error during signup'}), 500


# ✅ Login Route with JWT
@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()

        if not data.get('email') or not data.get('password'):
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        email = data['email'].lower().strip()
        user = users_collection.find_one({'email': email})

        if not user or not bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

        # ✅ Create JWT token (valid for 24 hours)
        token = jwt.encode({
            'user_id': str(user['_id']),
            'email': user['email'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': str(user['_id']),
                'firstName': user.get('firstName', ''),
                'lastName': user.get('lastName', ''),
                'email': user['email']
            }
        }), 200

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Server error during login'}), 500


# ✅ Example Protected Route
@app.route('/api/profile', methods=['GET'])
def profile():
    token = request.headers.get('Authorization')

    if not token:
        return jsonify({'success': False, 'message': 'Missing token'}), 401

    try:
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user = users_collection.find_one({'email': decoded['email']})

        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        return jsonify({
            'success': True,
            'user': {
                'id': str(user['_id']),
                'firstName': user.get('firstName'),
                'lastName': user.get('lastName'),
                'email': user.get('email')
            }
        }), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'success': False, 'message': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'success': False, 'message': 'Invalid token'}), 401


if __name__ == '__main__':
    app.run(debug=True, port=5000)

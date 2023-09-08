import json
import datetime


class FileStorage:
    def __init__(self, file_name):
        self.fine_name = file_name
        self.history = {}

#    def save(self, data):
#        self.history.update(data)
#        with open(self.fine_name, 'w', newline='') as f:
#            json.dump(self.history, f)

# solve duplicate user
    def save(self, data):
        # Load existing data
        existing_data = self.load()

        # Update existing data with new data
        existing_data.update(data)

        # Save the updated data to the file
        with open(self.fine_name, 'w', newline='') as f:
            json.dump(existing_data, f)
###

    def load(self):
        with open(self.fine_name, newline='') as jsonfile:
            data = json.load(jsonfile)
        self.history = data
        return self.history


class MongoStorage:
    def __init__(self, db):
        self.db = db

#    def save(self, data):
#        user_id, api_key = list(data.items())[0]
#        self.db['api_key'].update_one({
#            'user_id': user_id
#        }, {
#            '$set': {
#                'user_id': user_id,
#                'api_key': api_key,
#                'created_at': datetime.datetime.utcnow()
#            }
#        }, upsert=True)

# solve duplicate user
    def save(self, data):
        user_id, api_key = list(data.items())[0]

        # Check if data with the same user_id already exists in the database
        existing_data = self.db['api_key'].find_one({'user_id': user_id})

        if existing_data is None:
            # Data doesn't exist, insert it
            self.db['api_key'].insert_one({
                'user_id': user_id,
                'api_key': api_key,
                'created_at': datetime.datetime.utcnow()
            })
        else:
            # Data already exists, update it
            self.db['api_key'].update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        'api_key': api_key,
                        'created_at': datetime.datetime.utcnow()
                    }
                }
            )
###

    def load(self):
        data = list(self.db['student_id'].find())
        res = {}
        for i in range(len(data)):
            res[data[i]['user_id']] = data[i]['student_id']
        return res


class Storage:
    def __init__(self, storage):
        self.storage = storage

    def save(self, data):
        self.storage.save(data)

    def load(self):
        return self.storage.load()
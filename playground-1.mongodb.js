const { MongoClient } = require('mongodb');

const uri = 'mongodb+srv://jong0092:jong0092@cluster0.zr35dpg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';
const client = new MongoClient(uri);

async function addTestCredentials() {
    try {
        await client.connect();
        const db = client.db('Login_Credentials');
        const collection = db.collection('Login_Collection');

        const testStudents = { alice: '123', bob: 'qwerty' };
        const testTeachers = { mrsmith: 'abc', msjones: 'xyz' };

        // Insert students
        await collection.insertMany(
            Object.entries(testStudents).map(([username, password]) => ({
                username,
                password,
                userType: 'student'
            }))
        );

        // Insert teachers
        await collection.insertMany(
            Object.entries(testTeachers).map(([username, password]) => ({
                username,
                password,
                userType: 'teacher'
            }))
        );

        console.log('Test credentials added.');
    } finally {
        await client.close();
    }
}

addTestCredentials().catch(console.error);
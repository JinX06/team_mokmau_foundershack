const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true })); // For parsing form data
app.engine("html", require("ejs").renderFile);
app.set("views", __dirname + "/views");
app.set("view engine", "html");

// Testing users credentials (this should ideally be replaced by a database in a real-world app)
const testUsers = {
  jason: "123",
  bob: "qwerty",
  charlie: "12345",
};

// Routes
app.get('/', (req, res) => {
    res.render('index', { title: 'Home', message: 'Hello, world! IM FOO YU JUN' });
});

// About Us page
app.get('/about_us', (req, res) => {
    res.render('about_us', { title: 'About Us' });
});

// Plan & Pricing page
app.get('/plan_pricing', (req, res) => {
    res.render('plan_pricing', { title: 'Plan & Pricing' });
});

app.get('/login', (req, res) => {
    res.render('login', { title: 'Login' });a
});

// Handle login form
app.post("/login", (req, res) => {
  const { username, password } = req.body;

  // Check if the username and password are valid
  if (testUsers[username] && testUsers[username] === password) {
    // Redirect to the submission page if the credentials are correct
    return res.redirect("/submission");
  } else {
    // Show an error message if the credentials are incorrect
    return res.send(
      "<h1>Invalid username or password ❌</h1><a href='/login'>Try again</a>"
    );
  }
});

app.get('/submission', (req, res) => {
    res.render('submission', { title: 'File Submission' });
});

// Serve static files from the "public" directory
app.use(express.static(__dirname + '/public'));

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port http://localhost:${PORT}`);
});

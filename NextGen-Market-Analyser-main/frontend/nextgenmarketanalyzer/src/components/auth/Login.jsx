import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Container,
  Row,
  Col,
  Card,
  Form,
  Button,
  Spinner,
  Alert,
} from "react-bootstrap";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [loginMessage, setLoginMessage] = useState(null);

  const navigate = useNavigate();
  const API_BASE_URL = "http://localhost:5000/api";

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const newErrors = {};
    if (!validateEmail(email)) newErrors.email = "Enter a valid email";
    if (!password) newErrors.password = "Password is required";
    setErrors(newErrors);

    if (Object.keys(newErrors).length === 0) {
      setIsLoading(true);
      setLoginMessage(null);

      try {
        const res = await fetch(`${API_BASE_URL}/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        const data = await res.json();
        if (data.success) {
          localStorage.setItem("user", JSON.stringify(data.user));
          localStorage.setItem("isAuthenticated", "true");
          localStorage.setItem("token", data.token);
          setLoginMessage({
            type: "success",
            text: "Login successful! Redirecting...",
          });

          setTimeout(() => navigate("/dashboard"), 1500);
        } else {
          setLoginMessage({
            type: "danger",
            text: data.message || "Login failed.",
          });
        }
      } catch (err) {
        setLoginMessage({ type: "danger", text: "Network error. Try again." });
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <Container className="d-flex align-items-center justify-content-center min-vh-100 bg-light">
      <Row className="w-100" style={{ maxWidth: "420px" }}>
        <Col>
          <Card className="shadow-lg rounded-4 border-0 p-4">
            <h3 className="text-center mb-3 fw-bold text-success">
              Login to Your Account
            </h3>
            <p className="text-center text-muted">
              Access your personalized stock dashboard
            </p>

            {loginMessage && (
              <Alert variant={loginMessage.type} className="text-center">
                {loginMessage.text}
              </Alert>
            )}

            <Form onSubmit={handleSubmit}>
              <Form.Group className="mb-3">
                <Form.Label>Email Address</Form.Label>
                <Form.Control
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  isInvalid={!!errors.email}
                />
                <Form.Control.Feedback type="invalid">
                  {errors.email}
                </Form.Control.Feedback>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Password</Form.Label>
                <Form.Control
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  isInvalid={!!errors.password}
                />
                <Form.Control.Feedback type="invalid">
                  {errors.password}
                </Form.Control.Feedback>
              </Form.Group>

              <div className="d-flex justify-content-between align-items-center mb-3">
                <Form.Check type="checkbox" label="Remember me" />
                <a href="#" className="text-success small">
                  Forgot Password?
                </a>
              </div>

              <Button
                type="submit"
                variant="success"
                className="w-100 py-2 fw-bold"
                disabled={isLoading}
              >
                {isLoading ? <Spinner animation="border" size="sm" /> : "Login"}
              </Button>
            </Form>

            <div className="text-center mt-3">
              <span className="text-muted">Don't have an account?</span>{" "}
              <a href="/signup" className="text-success fw-bold">
                Sign Up
              </a>
            </div>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Login;

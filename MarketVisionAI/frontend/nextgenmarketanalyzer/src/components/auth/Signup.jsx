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

const Signup = () => {
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    phone: "",
    companyName: "",
    country: "",
    role: "",
  });
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [signupMessage, setSignupMessage] = useState(null);

  const navigate = useNavigate();
  const API_BASE_URL = "http://localhost:5000/api";

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  const handleChange = (e) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    const newErrors = {};
    if (!formData.firstName) newErrors.firstName = "First name is required";
    if (!formData.lastName) newErrors.lastName = "Last name is required";
    if (!validateEmail(formData.email)) newErrors.email = "Enter a valid email";
    if (formData.password.length < 6)
      newErrors.password = "Password must be at least 6 characters";

    setErrors(newErrors);

    if (Object.keys(newErrors).length === 0) {
      setIsLoading(true);
      setSignupMessage(null);

      try {
        const res = await fetch(`${API_BASE_URL}/signup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        });
        const data = await res.json();

        if (data.success) {
          setSignupMessage({
            type: "success",
            text: "Signup successful! Redirecting...",
          });
          setTimeout(() => navigate("/"), 1500);
        } else {
          setSignupMessage({
            type: "danger",
            text: data.message || "Signup failed.",
          });
        }
      } catch (err) {
        setSignupMessage({ type: "danger", text: "Network error. Try again." });
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <Container className="d-flex align-items-center justify-content-center min-vh-100 bg-light">
      <Row className="w-100" style={{ maxWidth: "600px" }}>
        <Col>
          <Card className="shadow-lg rounded-4 border-0 p-4">
            <h3 className="text-center mb-3 fw-bold text-success">
              Create an Account
            </h3>
            <p className="text-center text-muted">
              Join us and manage your stock portfolio smarter
            </p>

            {signupMessage && (
              <Alert variant={signupMessage.type} className="text-center">
                {signupMessage.text}
              </Alert>
            )}

            <Form onSubmit={handleSubmit}>
              <Row className="mb-3">
                <Col>
                  <Form.Control
                    placeholder="First Name"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                    isInvalid={!!errors.firstName}
                  />
                  <Form.Control.Feedback type="invalid">
                    {errors.firstName}
                  </Form.Control.Feedback>
                </Col>
                <Col>
                  <Form.Control
                    placeholder="Last Name"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                    isInvalid={!!errors.lastName}
                  />
                  <Form.Control.Feedback type="invalid">
                    {errors.lastName}
                  </Form.Control.Feedback>
                </Col>
              </Row>

              <Form.Control
                className="mb-3"
                placeholder="Email Address"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                isInvalid={!!errors.email}
              />
              <Form.Control.Feedback type="invalid">
                {errors.email}
              </Form.Control.Feedback>

              <Form.Control
                className="mb-3"
                placeholder="Password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                isInvalid={!!errors.password}
              />
              <Form.Control.Feedback type="invalid">
                {errors.password}
              </Form.Control.Feedback>

              <Form.Control
                className="mb-3"
                placeholder="Phone (Optional)"
                name="phone"
                value={formData.phone}
                onChange={handleChange}
              />
              <Form.Control
                className="mb-3"
                placeholder="Company Name"
                name="companyName"
                value={formData.companyName}
                onChange={handleChange}
              />
              <Row className="mb-3">
                <Col>
                  <Form.Control
                    placeholder="Country"
                    name="country"
                    value={formData.country}
                    onChange={handleChange}
                  />
                </Col>
                <Col>
                  <Form.Control
                    placeholder="Role"
                    name="role"
                    value={formData.role}
                    onChange={handleChange}
                  />
                </Col>
              </Row>

              <Button
                type="submit"
                variant="success"
                className="w-100 py-2 fw-bold"
                disabled={isLoading}
              >
                {isLoading ? (
                  <Spinner animation="border" size="sm" />
                ) : (
                  "Sign Up"
                )}
              </Button>
            </Form>

            <div className="text-center mt-3">
              <span className="text-muted">Already have an account?</span>{" "}
              <a href="/" className="text-success fw-bold">
                Login
              </a>
            </div>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Signup;

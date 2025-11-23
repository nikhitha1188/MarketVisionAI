import React, { useState, useEffect, useRef } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Form,
  Button,
  ListGroup,
  Table,
  Spinner,
} from "react-bootstrap";

const StockEvaluator = () => {
  const [stock, setStock] = useState("");
  const [stocksList, setStocksList] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryText, setSummaryText] = useState(""); // live streaming summary
  const summaryRef = useRef(""); // buffer for typewriter

  // Fetch stock symbols
  useEffect(() => {
    const fetchStocks = async () => {
      try {
        const token = localStorage.getItem("token");
        const res = await fetch("http://127.0.0.1:8000/stocks", {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        const data = await res.json();
        setStocksList(data.stocks || []);
      } catch (err) {
        console.error("Failed to load stocks list:", err);
        setStocksList(["AAPL", "MSFT", "GOOGL"]);
      }
    };
    fetchStocks();
  }, []);

  // Typewriter effect
  const typeWriterEffect = (textChunk) => {
    let i = 0;
    const interval = setInterval(() => {
      if (i < textChunk.length) {
        summaryRef.current += textChunk[i];
        setSummaryText(summaryRef.current);
        i++;
      } else {
        clearInterval(interval);
      }
    }, 20); // speed of typing
  };

  // Streaming summary
  const handleEvaluateStream = async (selectedStock) => {
    if (!selectedStock) return;
    setLoading(true);
    setEvaluation(null);
    setSummaryText("");

    // Queue for chunks & typing flag
    let chunkQueue = [];
    let typing = false;

    // Typewriter effect
    const typeWriterEffect = async (text) => {
      typing = true;
      for (let i = 0; i < text.length; i++) {
        setSummaryText((prev) => prev + text[i]);
        await new Promise((r) => setTimeout(r, 10));
      }
      typing = false;
      if (chunkQueue.length > 0) {
        const nextChunk = chunkQueue.shift();
        typeWriterEffect(nextChunk);
      }
    };

    try {
      const token = localStorage.getItem("token");
      // 1️⃣ Fetch stock evaluation
      const res = await fetch(
        `http://127.0.0.1:8000/evaluate/${selectedStock}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const data = await res.json();
      setEvaluation(data);

      setSummaryLoading(true);

      // 2️⃣ Stream feedback summary
      const streamRes = await fetch(
        "http://127.0.0.1:8000/send-feedback-stream",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" ,
          Authorization: `Bearer ${token}`
        },
          body: JSON.stringify(data),
        }
      );

      const reader = streamRes.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = ""; // define buffer here

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          // Split by newline for SSE
          let lines = buffer.split("\n\n");
          buffer = lines.pop(); // keep incomplete line

          for (let line of lines) {
            if (line.startsWith("data:")) {
              const jsonData = line.replace("data:", "").trim();
              try {
                const content = JSON.parse(jsonData);
                chunkQueue.push(content);
                if (!typing) typeWriterEffect(chunkQueue.shift());
              } catch (err) {
                console.error("JSON parse error:", err);
              }
            }
          }
        }
      }

      // Flush any remaining line
      if (buffer.startsWith("data:")) {
        const jsonData = buffer.replace("data:", "").trim();
        try {
          const content = JSON.parse(jsonData);
          chunkQueue.push(content);
          if (!typing) typeWriterEffect(chunkQueue.shift());
        } catch (err) {
          console.error("JSON parse error:", err);
        }
      }

      setSummaryLoading(false);
    } catch (err) {
      console.error(err);
      alert("Error evaluating stock: " + err.message);
      setSummaryLoading(false);
    }

    setLoading(false);
  };



  const safe = (val) => (val !== undefined && val !== null ? val : "-");

  const renderSummary = (summary) => {
    if (!summary) return null;
    const lines = summary.split("\n").filter((line) => line.trim() !== "");
    const bulletLines = [];
    const normalLines = [];

    lines.forEach((line) => {
      if (line.startsWith("*")) bulletLines.push(line.replace(/^\*\s*/, ""));
      else normalLines.push(line);
    });

    const parseBold = (line, key) => {
      const boldRegex = /\*\*(.*?)\*\*/g;
      const parts = [];
      let lastIndex = 0;
      let match;

      while ((match = boldRegex.exec(line)) !== null) {
        if (match.index > lastIndex) parts.push(line.slice(lastIndex, match.index));
        parts.push(<strong key={key + match.index}>{match[1]}</strong>);
        lastIndex = match.index + match[0].length;
      }
      if (lastIndex < line.length) parts.push(line.slice(lastIndex));
      return parts;
    };

    return (
      <>
        {normalLines.map((line, idx) => (
          <p key={"p" + idx} className="text-muted fs-6 mb-1">
            {parseBold(line, idx)}
          </p>
        ))}
        {bulletLines.length > 0 && (
          <ul className="text-muted fs-6 mb-2">
            {bulletLines.map((line, idx) => (
              <li key={"li" + idx}>{parseBold(line, idx)}</li>
            ))}
          </ul>
        )}
      </>
    );
  };

  return (
    <Container fluid className="p-3">
      <Row className="gy-4" style={{ minHeight: "100%" }}>
        {/* Sidebar */}
        <Col lg={2} md={3} sm={12}>
          <div
            style={{
              position: "sticky",
              top: "1rem",
              maxHeight: "calc(100vh - 2rem)",
              overflowY: "auto",
            }}
          >
            <Card
              className="shadow-sm rounded-4 border-0 d-flex flex-column h-100"
              style={{ background: "#f1f3f5" }}
            >
              <Card.Header className="bg-success text-white fw-bold rounded-top-4">
                Frequently Searched Stocks
              </Card.Header>
              <ListGroup
                variant="flush"
                className="flex-grow-1"
                style={{ overflowY: "auto" }}
              >
                {stocksList.map((s, idx) => (
                  <ListGroup.Item
                    key={idx}
                    className="py-3 fw-semibold hover-shadow"
                    style={{ cursor: "pointer" }}
                    onClick={() => {
                      setStock(s);
                      handleEvaluateStream(s);
                    }}
                  >
                    {s}
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Card>
          </div>
        </Col>

        {/* Evaluator */}
        <Col lg={9} md={8} sm={12}>
          <Card
            className="shadow-sm mb-4 rounded-4 border-0 p-3"
            style={{ background: "#f8f9fa" }}
          >
            <Form>
              <Row className="align-items-center g-3">
                <Col md={8} sm={12}>
                  <Form.Control
                    type="text"
                    placeholder="Enter stock symbol (e.g., AAPL)"
                    value={stock}
                    onChange={(e) => setStock(e.target.value.toUpperCase())}
                    className="py-3 rounded-3 border-success"
                  />
                </Col>
                <Col md={4} sm={12}>
                  <Button
                    variant="success"
                    className="w-100 py-3 rounded-3 fw-bold"
                    onClick={() => handleEvaluateStream(stock)}
                    disabled={loading} // ✅ keep this so user can’t spam while initial evaluation is happening
                  >
                    🚀 Evaluate
                  </Button>
                </Col>
              </Row>
            </Form>
          </Card>

          {/* Evaluation Result */}
          {evaluation && (
            <>
              <Row className="gy-4">
                {/* Key Metrics */}
                <Col lg={6} md={12}>
                  <Card
                    className="shadow-sm p-3 rounded-4 border-0 mb-4 h-100"
                    style={{ background: "#ffffff" }}
                  >
                    <h5 className="fw-bold mb-3 text-success">
                      📊 Key Metrics
                    </h5>
                    <Table responsive striped bordered hover className="mb-0">
                      <tbody>
                        {evaluation.values &&
                          Object.entries(evaluation.values).map(
                            ([key, value]) => (
                              <tr key={key}>
                                <th>{key.replace(/([A-Z])/g, " $1")}</th>
                                <td>{safe(value)}</td>
                              </tr>
                            )
                          )}
                      </tbody>
                    </Table>
                  </Card>
                </Col>
                {/* Right Column */}
                <Col lg={6} md={12}>
                  {/* Recommendation card */}
                  {evaluation.recommendation && (
                    <Card
                      className="shadow-sm rounded-2 border-0 mb-2"
                      style={{
                        background: "#e6ffed",
                        padding: "0.5rem 1rem", // smaller padding
                        lineHeight: "1.2", // tighter line spacing
                      }}
                    >
                      <h5
                        className="fw-bold mb-1 text-success"
                        style={{ fontSize: "1rem" }}
                      >
                        🏆 Recommendation
                      </h5>
                      <p className="mb-0" style={{ fontSize: "0.95rem" }}>
                        {evaluation.recommendation}
                      </p>
                    </Card>
                  )}

                  {/* Detailed Feedback card */}
                  {evaluation.feedback && (
                    <Card
                      className="shadow-sm p-3 rounded-4 border-0"
                      style={{ background: "#f8f9fa" }}
                    >
                      <h5 className="fw-bold mb-2 text-success">
                        📝 Detailed Feedback
                      </h5>
                      <ul className="list-unstyled mb-0">
                        {Object.entries(evaluation.feedback).map(
                          ([key, value]) => (
                            <li key={key} className="mb-1">
                              <strong>
                                {key.replace(/([A-Z])/g, " $1")}:{" "}
                              </strong>{" "}
                              {value}
                            </li>
                          )
                        )}
                      </ul>
                    </Card>
                  )}
                </Col>
              </Row>

              {/* Recommendation */}

              <Card
                className="shadow-sm p-3 rounded-4 border-0 mb-4 h-100"
                style={{ background: "#ffffff" }}
              >
                <h5 className="fw-bold mb-3 text-success">📋 Summary</h5>
                {summaryLoading ? (
                  <div className="text-center py-5">
                    <Spinner animation="border" variant="success" role="status">
                      <span className="visually-hidden">
                        Generating summary...
                      </span>
                    </Spinner>
                    <p className="mt-2 text-muted">Generating summary...</p>
                  </div>
                ) : (
                  renderSummary(summaryText)
                )}
              </Card>
            </>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default StockEvaluator;

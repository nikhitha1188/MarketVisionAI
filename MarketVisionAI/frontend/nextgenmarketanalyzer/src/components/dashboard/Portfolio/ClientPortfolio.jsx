import React from "react";
import { PieChart, Pie, Cell, Tooltip } from "recharts";
import "./ClientPortfolio.css";

const ClientPortfolio = ({ clients }) => {
  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#A28DFF"];

  return (
    <div className="client-portfolio">
      {clients.map((client, index) => (
        <div key={client.clientId} className="client-card">
          <h2>Client ID: {client.clientId}</h2>
          <h3>Currency: {client.currency}</h3>

          {client.funds.map((fund, fundIndex) => (
            <div key={fund.fundCode} className="fund-card">
              <h4>Fund Code: {fund.fundCode}</h4>
              <p>Amount: ₹{fund.amount.toLocaleString()}</p>

              <div className="chart-container">
                <h5>Sector Diversification</h5>
                <PieChart width={300} height={300}>
                  <Pie
                    data={Object.entries(fund.sectors).map(([name, value]) => ({
                      name,
                      value,
                    }))}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    label
                  >
                    {Object.entries(fund.sectors).map((_, sectorIndex) => (
                      <Cell
                        key={`cell-${sectorIndex}`}
                        fill={COLORS[sectorIndex % COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default ClientPortfolio;
import React, { useState, useEffect } from "react";
import relativeTime from "dayjs/plugin/relativeTime";
import utc from "dayjs/plugin/utc";

dayjs.extend(relativeTime);
dayjs.extend(utc);

export default function DefaultWidget()
{
    const [loading, setLoading] = useState(false);

    const [teamA, setTeamA] = useState(["", "", "", "", ""]);
    const [teamB, setTeamB] = useState(["", "", "", "", ""]);

    const [results, setResults] = useState([false, -1, -1]);

    const [mode, setMode] = useState(0);
    
    const handleChange = (i, newValue, team) =>
    {
        if (team === "A")
        {
            const updated = [...teamA]
            updated[i] = newValue
            setTeamA(updated)
        }
        else
        {
            const updated = [...teamB]
            updated[i] = newValue
            setTeamB(updated)
        }
    }

    const handleDropdown = (e) => {
        setMode(parseInt(e.target.value));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        setLoading(true);
        try
        {   
            const fetchResults = async () => {
                const response = await fetch("/api/v1/predict/", {
                    method: "POST",
                    credentials: "same-origin",
                    headers: {
                    "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        team1: teamA,
                        team2: teamB,
                    }),
                });
                if (!response.ok) throw new Error(response.statusText);
                const data = await response.json();
                setResults(data.results);
                setLoading(false);
            }
            fetchResults();
        }
        catch (error)
        {
            console.error("Error fetching results:", error);
            setLoading(false);
        }
    }

    return (
        <div className="Interactable Widget">
            <form onSubmit={handleSubmit}>
                Enter Players On Team A:
                {teamA.map((val, i) => (
                    <input
                        key={i}
                        type="text"
                        value={val}
                        placeholder={`Player ${i + 1}`}
                        onChange={(e) => handleChange(i, e.target.value, "A")}
                    />
                ))}
                Enter Players On Team B:
                {teamB.map((val, i) => (
                    <input
                        key={i}
                        type="text"
                        value={val}
                        placeholder={`Player ${i + 1}`}
                        onChange={(e) => handleChange(i, e.target.value, "B")}
                    />
                ))}

                <label htmlFor="mode-select">Select Game Mode: </label>
                <select id="mode-select" value={mode} onChange={handleChange}>
                    <option value={0}>Quick Play</option>
                    <option value={1}>Standard</option>
                    <option value={2}>Ranked</option>
                </select>

                <p>Selected Mode: {mode}</p>
                
                {loading ? (
                    <p>Submitting... please wait</p>
                ) : (<button type="submit">
                    Submit
                </button>)
                }
            </form>

            {results[0] ? (
            <div>
                <p>{results[1] ? "Team A is favored to win" : "Team B is favored to win"}</p>
                <p>Chance Team A Will Win: {results[0]}%</p>
                <p>Chance Team B Will Win: {100 - results[0]}%</p>
            </div>
            ) : null}
        </div>
    );
}
import React, { useState } from "react";
import "./Abusive.css";
import profile from '../../assets/profile.webp'
function AbusiveDetect() {
  const [score, setScore] = useState("")
  const [textArea, settextArea] = useState("")
  const handleSubmit = async (e) => {
    e.preventDefault()
    const res = await fetch("http://127.0.0.1:8000/api/model/",{
      method:'POST',
      headers:{
        "Content-Type": "application/json"
      },
      body:JSON.stringify({
        text:textArea
      })
    })
    const resJson = await res.json()
    if(resJson.status = "ok"){
      setScore(resJson.score)
    }else if(resJson.status = "smw"){
      alert(resJson.message)
    }
  }


  return (
    <>
      <div className="cont">
        <div className="nav">
          <ul className="content">
            <div className="left">
              <li><a href="/">Home</a></li>
              <li><a href="/pricing">Pricing</a></li>
              <div className="dropdown">
                <li>Resources</li>
                <div className="dropdown-options">
                  <a href="#">GetText</a>
                  <a href="#">Blog</a>
                  <a href="#">About Us</a>
                </div>
              </div>
            </div>
            <div className="right">
              <a href='/'>Login</a>
              <img src={profile} alt="profile icon" />
            </div>
          </ul>
        </div>
        <div className="main">
         <b><h1 className="h">Hate Content detection</h1></b> 
          <textarea className="text" placeholder="Enter the text" onChange={(e)=>settextArea(e.target.value)} value={textArea}></textarea>
        </div>
          <button className="btn" type="submit" onClick={handleSubmit}>
            Submit
          </button>
        <div className="circle">
          <span>
            {score}
          </span>
        </div>
        <h3 className="footer">score</h3>
      </div>
    </>
  );
}

export default AbusiveDetect;

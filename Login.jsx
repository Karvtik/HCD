import React, { useState } from 'react';
import './Login.css';
import { useNavigate } from 'react-router-dom';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate()
  const handleEmailChange = (event) => {
    setEmail(event.target.value);
  };

  const handlePasswordChange = (event) => {
    setPassword(event.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await fetch("http://127.0.0.1:8000/api/login/",{
      method:'POST',
      headers:{
        "Content-Type": "application/json"
      },
      body:JSON.stringify({
        email:email,
        password:password
      })
    })
    const resJson = await res.json()
    if(resJson.status == "ok"){
      window.localStorage.setItem("user_id",resJson.user_id)
      navigate("/home")
    }else{
      alert(resJson.message)
    }
  }

  return (
    <>
      <div className="contain">
        <div className="box">
          <h1>Login</h1>
          <form>
            <div className="email align">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                placeholder="Enter Your Email"
                value={email}
                onChange={handleEmailChange}
              />
            </div>
            <div className="password align">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                placeholder="Enter Your Password"
                value={password}
                onChange={handlePasswordChange}
              />
            </div>
            <div className="extra">
              <div className="remember">
                <input type="checkbox" id="checkbox" />
                <label htmlFor="checkbox">Remember me</label>
              </div>
              <a href="/forgotpassword">Forgot Password?</a>
            </div>
            <div className="submit">
              <button type="submit" onClick={handleSubmit}>Login</button>
            </div>
          </form>
          <div className="register">
            Create New Account
            <a href="/signup">SignUp</a>
          </div>
        </div>
      </div>
    </>
  );
}

export default Login;

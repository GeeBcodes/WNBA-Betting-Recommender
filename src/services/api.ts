import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000', // backend API URL
  headers: {
    'Content-Type': 'application/json',
    // can add other common headers here, like Authorization tokens if needed
  },
});

// can also add interceptors for request/response handling globally
// For example, to handle errors or add auth tokens to every request:
/*
apiClient.interceptors.request.use(
  (config) => {
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle errors globally
    // For example, redirect to login if 401 Unauthorized
    // if (error.response && error.response.status === 401) {
    //   // window.location.href = '/login';
    // }
    return Promise.reject(error);
  }
);
*/

export default apiClient; 
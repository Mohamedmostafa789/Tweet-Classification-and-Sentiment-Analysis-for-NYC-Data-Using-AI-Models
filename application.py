import React, { useState, useEffect } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithCustomToken, signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, collection, onSnapshot, query, limit, orderBy } from 'firebase/firestore';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './shadcn/ui/card';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

// Define the shape of a single tweet result
const TweetResult = ({ tweet }) => {
  const sentimentColors = {
    Positive: 'bg-green-100',
    Negative: 'bg-red-100',
    Neutral: 'bg-yellow-100'
  };

  const sentimentTextColors = {
    Positive: 'text-green-800',
    Negative: 'text-red-800',
    Neutral: 'text-yellow-800'
  };

  return (
    <div className={`p-4 rounded-lg shadow mb-2 ${sentimentColors[tweet.sentiment]}`}>
      <p className="text-gray-800 font-medium">"{tweet.tweet_text}"</p>
      <div className="flex items-center mt-2 text-sm">
        <span className={`font-bold ${sentimentTextColors[tweet.sentiment]}`}>
          {tweet.sentiment}
        </span>
        <span className="ml-4 text-gray-500">Emotion: {tweet.emotion}</span>
      </div>
    </div>
  );
};

// Main App component
const App = () => {
  // State variables for Firebase, user info, data, and UI status
  const [db, setDb] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [userId, setUserId] = useState(null);
  const [tweets, setTweets] = useState([]);
  const [sentimentData, setSentimentData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // --- Firebase Initialization and Authentication ---
  // This effect runs once to set up Firebase and authentication listener.
  useEffect(() => {
    try {
      const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : null;
      if (!firebaseConfig) {
        throw new Error("Firebase config not found.");
      }

      const app = initializeApp(firebaseConfig);
      const dbInstance = getFirestore(app);
      const authInstance = getAuth(app);

      setDb(dbInstance);

      // Listen for auth state changes to get the user ID
      const unsubscribe = onAuthStateChanged(authInstance, async (user) => {
        if (user) {
          setUserId(user.uid);
          setIsAuthReady(true);
        } else {
          // If no user is logged in, use the custom auth token if available
          const token = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;
          if (token) {
            try {
              await signInWithCustomToken(authInstance, token);
            } catch (e) {
              console.error("Error signing in with custom token:", e);
              await signInAnonymously(authInstance);
            }
          } else {
            await signInAnonymously(authInstance);
          }
        }
      });

      return () => unsubscribe();
    } catch (e) {
      console.error("Firebase setup error:", e);
      setError("Failed to initialize the application. Please check the console for details.");
      setLoading(false);
    }
  }, []);

  // --- Data Fetching and Real-time Updates ---
  // This effect runs whenever the auth state is ready and the db instance is available.
  useEffect(() => {
    if (!isAuthReady || !db) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Use the provided app ID to construct the correct path
      const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
      const collectionRef = collection(db, `artifacts/${appId}/public/data/sentiment_results`);
      
      // Create a query to get a limited number of the latest tweets.
      // Note: In a production app, Firestore needs an index for this query.
      const q = query(collectionRef, orderBy('timestamp', 'desc'), limit(50));

      // Use onSnapshot for real-time updates
      const unsubscribe = onSnapshot(q, (querySnapshot) => {
        const fetchedTweets = [];
        const sentimentCounts = { Positive: 0, Negative: 0, Neutral: 0 };
        
        querySnapshot.forEach((doc) => {
          const data = doc.data();
          fetchedTweets.push({ id: doc.id, ...data });
          
          // Count sentiments for the pie chart
          if (data.sentiment) {
            sentimentCounts[data.sentiment] = (sentimentCounts[data.sentiment] || 0) + 1;
          }
        });
        
        setTweets(fetchedTweets);
        setSentimentData([
          { name: 'Positive', value: sentimentCounts.Positive, color: '#4CAF50' },
          { name: 'Negative', value: sentimentCounts.Negative, color: '#F44336' },
          { name: 'Neutral', value: sentimentCounts.Neutral, color: '#FFC107' },
        ]);
        setLoading(false);
      }, (err) => {
        console.error("Error fetching data:", err);
        setError("Failed to fetch data from the database. Please try again.");
        setLoading(false);
      });

      // Cleanup listener on unmount
      return () => unsubscribe();
    } catch (e) {
      console.error("Error setting up data listener:", e);
      setError("An unexpected error occurred while setting up the data listener.");
      setLoading(false);
    }
  }, [isAuthReady, db]);

  // Handle various states of the app (loading, error, content)
  const renderContent = () => {
    if (error) {
      return (
        <Card className="w-full max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle>Error</CardTitle>
            <CardDescription>An error occurred while loading the data.</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-red-500">{error}</p>
          </CardContent>
        </Card>
      );
    }

    if (loading) {
      return (
        <div className="flex items-center justify-center p-8">
          <svg className="animate-spin h-8 w-8 text-gray-900" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="ml-3 text-lg font-medium text-gray-700">Loading data...</p>
        </div>
      );
    }
    
    // Check if there are any tweets to display
    const hasData = tweets.length > 0;

    return (
      <div className="flex flex-col md:flex-row gap-8">
        {/* Sentiment Overview Card */}
        <Card className="flex-1 min-w-0 md:min-w-[400px]">
          <CardHeader>
            <CardTitle>Sentiment Overview</CardTitle>
            <CardDescription>Distribution of sentiments from the analyzed tweets.</CardDescription>
          </CardHeader>
          <CardContent className="h-64">
            {hasData ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sentimentData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    labelLine={false}
                    isAnimationActive={false}
                  >
                    {sentimentData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                <p>No data to display. Please run the Python script to populate the database.</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Tweets Card */}
        <Card className="flex-1 min-w-0">
          <CardHeader>
            <CardTitle>Recent Tweets</CardTitle>
            <CardDescription>A real-time feed of the latest analyzed tweets.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
              {tweets.length > 0 ? (
                tweets.map((tweet) => (
                  <TweetResult key={tweet.id} tweet={tweet} />
                ))
              ) : (
                <div className="text-gray-500">
                  <p>No tweets found. The database may be empty or still processing.</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };
  
  return (
    <div className="p-8 font-sans bg-gray-50 min-h-screen">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">NYC Tweet Analysis Dashboard</h1>
        {userId && (
          <div className="text-sm text-gray-600">
            User ID: <span className="font-mono text-gray-800 break-all">{userId}</span>
          </div>
        )}
      </div>
      {renderContent()}
    </div>
  );
};

export default App;

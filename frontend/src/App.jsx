import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Legend,
  Area,
  AreaChart,
} from "recharts";
import {
  Upload,
  Search,
  TrendingUp,
  MessageSquare,
  Star,
  FileText,
  Download,
  RefreshCw,
  Filter,
  Eye,
  ThumbsUp,
  ThumbsDown,
  AlertCircle,
  CheckCircle,
} from "lucide-react";

const App = () => {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reviews, setReviews] = useState([]); // State untuk data dashboard yang akan diambil dari backend

  const [dashboardStats, setDashboardStats] = useState({
    totalReviews: "N/A",
    positiveReviews: "N/A",
    negativeReviews: "N/A",
    neutralReviews: "N/A",
    averageRating: "N/A",
  });
  const [sentimentData, setSentimentData] = useState([]); // Untuk PieChart
  const [sentimentInsights, setSentimentInsights] = useState({
    mostPositivePhrases: [],
    mostNegativePhrases: [],
    categoryAnalysis: [],
  }); // Fetch dashboard data on component mount and after successful upload

  const fetchDashboardData = async () => {
    try {
      const statsResponse = await fetch(
        "http://localhost:5000/api/dashboard-stats"
      );
      const statsData = await statsResponse.json();
      if (statsResponse.ok) {
        setDashboardStats(statsData); // Update sentimentData for PieChart based on fetched stats
        setSentimentData([
          {
            name: "Positive",
            value:
              typeof statsData.positiveReviews === "number"
                ? statsData.positiveReviews
                : 0,
            color: "#10B981",
          },
          {
            name: "Negative",
            value:
              typeof statsData.negativeReviews === "number"
                ? statsData.negativeReviews
                : 0,
            color: "#EF4444",
          },
          {
            name: "Neutral",
            value:
              typeof statsData.neutralReviews === "number"
                ? statsData.neutralReviews
                : 0,
            color: "#6B7280",
          },
        ]);
      } else {
        console.error("Failed to fetch dashboard stats:", statsData.error);
      }

      const insightsResponse = await fetch(
        "http://localhost:5000/api/sentiment-insights"
      );
      const insightsData = await insightsResponse.json();
      if (insightsResponse.ok) {
        setSentimentInsights(insightsData);
        console.log(insightsData);
      } else {
        console.error(
          "Failed to fetch sentiment insights:",
          insightsData.error
        );
      } // Fetch latest reviews

      const reviewsResponse = await fetch(
        "http://localhost:5000/api/latest-reviews"
      );
      const reviewsData = await reviewsResponse.json();
      if (reviewsResponse.ok) {
        setReviews(
          reviewsData.map((review) => ({
            id: review.id,
            text: review.text,
            sentiment: review.sentiment,
            confidence: review.confidence,
            keywords: review.keywords || [],
            rating: review.rating || 0,
          }))
        );
      } else {
        console.error("Failed to fetch latest reviews:", reviewsData.error);
      }
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setLoading(true);
      setAnalysisResults(null);
      setReviews([]);

      const formData = new FormData();
      formData.append("reviewsFile", file);

      try {
        const response = await fetch(
          "http://localhost:5000/api/upload-reviews",
          {
            method: "POST",
            body: formData,
          }
        );

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to analyze reviews");
        }

        const data = await response.json();
        console.log("Backend response:", data);

        setAnalysisResults(data.summary); // Reviews akan di-fetch ulang oleh fetchDashboardData // setReviews(data.reviews.map(review => ({ ... }))); // Ini bisa dihapus jika fetchDashboardData memuatnya // Setelah upload berhasil, refresh data dashboard dan reviews
        fetchDashboardData();
      } catch (error) {
        console.error("Error uploading file:", error);
        alert(`Error: ${error.message}`);
      } finally {
        setLoading(false);
      }
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case "positive":
        return "text-green-600 bg-green-100";
      case "negative":
        return "text-red-600 bg-red-100";
      default:
        return "text-gray-600 bg-gray-100";
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case "positive":
        return <ThumbsUp className="w-4 h-4" />;
      case "negative":
        return <ThumbsDown className="w-4 h-4" />;
      default:
        return <AlertCircle className="w-4 h-4" />;
    }
  };

  const Sidebar = () => (
    <div className="w-64 bg-white shadow-lg h-full">
      <div className="p-6">
        <h1 className="text-2xl font-bold text-gray-800 ml-10 mb-8">
          E-Commerce Sentiment Analyzer
        </h1>
        <nav className="space-y-2">
          {[
            { id: "dashboard", label: "Dashboard", icon: TrendingUp },
            { id: "analyzer", label: "Review Analyzer", icon: Search },
            { id: "insights", label: "Sentiment Insights", icon: Eye },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                activeTab === item.id
                  ? "bg-blue-100 text-blue-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              {React.createElement(item.icon, { className: "w-5 h-5" })}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </div>
    </div>
  );

  const MetricCard = ({ title, value, icon, color = "blue" }) => (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600">{title}</p>{" "}
          <p className="text-2xl font-bold text-gray-800">{value}</p>{" "}
        </div>
        <div className={`p-3 rounded-full bg-${color}-100`}>
          {React.createElement(icon, {
            className: `w-6 h-6 text-${color}-600`,
          })}
        </div>
      </div>
    </div>
  );

  const Dashboard = () => (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800">Dashboard</h2>
      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Reviews"
          value={dashboardStats.totalReviews}
          icon={MessageSquare}
          color="blue"
        />
        <MetricCard
          title="Positive Reviews"
          value={dashboardStats.positiveReviews}
          icon={ThumbsUp}
          color="green"
        />
        <MetricCard
          title="Negative Reviews"
          value={dashboardStats.negativeReviews}
          icon={ThumbsDown}
          color="red"
        />
        <MetricCard
          title="Average Confidence"
          value={
            typeof dashboardStats.averageRating === "number"
              ? dashboardStats.averageRating.toFixed(1)
              : "N/A"
          }
          icon={Star}
          color="yellow"
        />
      </div>
      {/* Charts */}
      <div className="">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Sentiment Distribution
          </h3>
          {sentimentData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={sentimentData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                >
                  {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center text-gray-500 py-10">
              No sentiment data available.
            </div>
          )}
        </div>
      </div>
      {/* Top Keywords */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        {" "}
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Top Keywords
        </h3>
        <div className="flex flex-wrap gap-2">
          {sentimentInsights.mostPositivePhrases.map((item, index) => (
            <span
              key={index}
              className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm"
            >
              {item.phrase} ({item.count})
            </span>
          ))}
        </div>
        {/*
        TODO: Fetch top keywords from backend.
        Example:
        const [topKeywords, setTopKeywords] = useState([]);
        useEffect(() => {
        const fetchKeywords = async () => {
              try {
                const response = await fetch('http://localhost:5000/api/sentiment-insights');
                if (!response.ok) throw new Error('Failed to fetch insights');
                const data = await response.json();
                // Assuming backend returns a 'topKeywords' array directly or within 'insights'
                setTopKeywords(data.topKeywords || []);
              } catch (error) {
                console.error('Error fetching top keywords:', error);
              }
            };
            fetchKeywords();
          }, []);
        */}
      </div>
    </div>
  );

  const ReviewAnalyzer = () => (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800">Review Analyzer</h2>
      {/* Upload Section */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Upload Reviews Dataset
        </h3>
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600 mb-4">
            Upload your CSV file with reviews
          </p>
          <input
            type="file"
            accept=".csv, .ft.txt" // Tambahkan .ft.txt jika ingin mendukungnya langsung
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="bg-blue-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-blue-700 transition-colors"
          >
            Choose File
          </label>
          {selectedFile && (
            <p className="mt-2 text-sm text-gray-700">
              Selected file:
              <span className="font-medium">{selectedFile.name}</span>
            </p>
          )}
        </div>
      </div>
      {/* Analysis Results */}
      {analysisResults && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Analysis Results
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-sm text-gray-600">Total Reviews</p>
              <p className="text-2xl font-bold text-gray-800">
                {analysisResults.totalReviews}
              </p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <p className="text-sm text-green-600">Positive</p>
              <p className="text-2xl font-bold text-green-800">
                {analysisResults.positiveCount}
              </p>
            </div>
            <div className="bg-red-50 p-4 rounded-lg">
              <p className="text-sm text-red-600">Negative</p>
              <p className="text-2xl font-bold text-red-800">
                {analysisResults.negativeCount}
              </p>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg">
              <p className="text-sm text-blue-600">Avg Confidence</p>
              <p className="text-2xl font-bold text-blue-800">
                {(analysisResults.avgConfidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          <button className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors flex items-center space-x-2">
            <Download className="w-4 h-4" /> <span>Export Results</span>
          </button>
        </div>
      )}
      {/* Review List */}
      {reviews.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Review Analysis
          </h3>
          <div className="space-y-4">
            {reviews.map((review) => (
              <div
                key={review.id}
                className="border border-gray-200 rounded-lg p-4"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${getSentimentColor(
                        review.sentiment
                      )}`}
                    >
                      {getSentimentIcon(review.sentiment)}
                      <span className="capitalize">{review.sentiment}</span>
                    </span>{" "}
                    <span className="text-sm text-gray-600">
                      Confidence: {(review.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <p className="text-gray-800 mb-2">{review.text}</p>
                <div className="flex flex-wrap gap-1">
                  {review.keywords.map((keyword, index) => (
                    <span
                      key={index}
                      className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {loading && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-center space-x-2">
            <RefreshCw className="w-5 h-5 animate-spin text-blue-600" />
            <span className="text-gray-600">Analyzing reviews...</span>
          </div>
        </div>
      )}
    </div>
  );
  const SentimentInsights = () => (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800">Sentiment Insights</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Most Positive Phrases
          </h3>
          <div className="space-y-2">
            {sentimentInsights.mostPositivePhrases.map((item, index) => (
              <div
                key={index}
                className="flex justify-between items-center p-2 bg-green-50 rounded"
              >
                <span className="text-green-700">{item.phrase}</span>
                <span className="text-sm text-green-600">{item.count}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Most Negative Phrases
          </h3>
          <div className="space-y-2">
            {sentimentInsights.mostNegativePhrases.map((item, index) => (
              <div
                key={index}
                className="flex justify-between items-center p-2 bg-red-50 rounded"
              >
                <span className="text-red-700">{item.phrase}</span>
                <span className="text-sm text-red-600">{item.count}</span>{" "}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Category Analysis
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {sentimentInsights.categoryAnalysis.map((item, index) => (
            <div key={index} className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-gray-800 mb-2">
                {item.category}
              </h4>
              <div className="flex items-center space-x-2 mb-1">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${item.positive}%` }}
                  ></div>
                </div>
                <span className="text-sm text-green-600">{item.positive}%</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-red-500 h-2 rounded-full"
                    style={{ width: `${item.negative}%` }}
                  ></div>
                </div>
                <span className="text-sm text-red-600">{item.negative}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case "dashboard":
        return <Dashboard />;
      case "analyzer":
        return <ReviewAnalyzer />;
      case "insights":
        return <SentimentInsights />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 overflow-auto">
        <div className="p-6">{renderContent()}</div>{" "}
      </div>
    </div>
  );
};

export default App;

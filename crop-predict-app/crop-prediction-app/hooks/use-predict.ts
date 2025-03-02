import { useState } from "react";

const API_URL = "http://localhost:8000/predict";


interface Prediction {
    crop: string;
    probability: number;
    percentage: string;
}
  
interface PredictResponse {
    predictions: Prediction[];
}
  
export const usePredict = () => {
  const [data, setData] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const predict = async (features: number[]) => {
    setLoading(true);
    setError(null);

    try {
        console.log(features)
        
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const result: PredictResponse = await response.json();
      setData(result);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return { data, loading, error, predict };
};

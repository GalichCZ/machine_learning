"use client"

import type React from "react"

import { useState } from "react"
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { usePredict } from "@/hooks/use-predict"

const PARAMETERS = [
  "N",
  "P",
  "K",
  "temperature",
  "humidity",
  "ph",
  "rainfall",
]

export default function CropPredictionApp() {
  const { data, loading, error, predict } = usePredict();
  const [inputValues, setInputValues] = useState(Array(7).fill(""))

  const handlePredict = () => {
    predict(inputValues.map(Number));
  };


  // Sample prediction data (static for this example)
  const predictionData = [
    { crop: "cotton", probability: 0.9758336249093642, percentage: "97.58%" },
    { crop: "maize", probability: 0.02216034001824768, percentage: "2.22%" },
    { crop: "watermelon", probability: 0.0012690149772054777, percentage: "0.13%" },
    { crop: "coffee", probability: 0.0005072635576409095, percentage: "0.05%" },
    { crop: "jute", probability: 0.00011980823361038744, percentage: "0.01%" },
  ]

  // Format data for the chart
  const chartData = data?.predictions?.map((item) => ({
    name: item.crop,
    probability: (item.probability * 100).toFixed(2),
    percentage: item.percentage,
  }))

  const handleInputChange = (index: number, value: string) => {
    const newValues = [...inputValues]
    newValues[index] = value
    setInputValues(newValues)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    handlePredict()
    console.log("Submitted values:", inputValues)
    // In a real app, you would send these values to an API
    // and update the prediction data based on the response
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-6">Crop Prediction App</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Input Parameters</h2>
          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {Array.from({ length: 7 }).map((_, index) => (
                <div key={index} className="space-y-2">
                  <Label htmlFor={`param-${index}`}>{PARAMETERS[index]}</Label>
                  <Input
                    id={PARAMETERS[index]}
                    type="number"
                    step="any"
                    placeholder={`Enter float value ${index + 1}`}
                    value={inputValues[index]}
                    onChange={(e) => handleInputChange(index, e.target.value)}
                    required
                  />
                </div>
              ))}
            </div>
            <Button type="submit" className="w-full">
              Submit
            </Button>
          </form>
        </Card>

        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
          <div className="h-[400px] w-full">
            <BarChart
              width={500}
              height={350}
              data={chartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              layout="vertical"
              className="mx-auto"
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" label={{ value: "Probability (%)", position: "bottom", offset: 0 }} />
              <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 14 }} />
              <Tooltip
                formatter={(value) => [`${value}%`, "Probability"]}
                labelFormatter={(label) => `Crop: ${label}`}
              />
              <Bar dataKey="probability" fill="hsl(var(--primary))" name="Probability (%)" />
            </BarChart>
          </div>

          <div className="mt-4">
            <h3 className="text-lg font-medium mb-2">Detailed Results:</h3>
            <ul className="space-y-1">
              {data?.predictions.map((item, index) => (
                <li key={index} className="flex justify-between">
                  <span className="capitalize">{item.crop}:</span>
                  <span className="font-medium">{item.percentage}</span>
                </li>
              ))}
            </ul>
          </div>
        </Card>
      </div>
    </div>
  )
}


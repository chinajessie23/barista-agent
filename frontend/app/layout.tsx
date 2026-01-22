import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Barista Agent",
  description: "AI-powered coffee shop ordering assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}

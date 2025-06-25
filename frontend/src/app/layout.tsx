'use client'; 

import { useState } from "react"; 
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { TrainingProvider } from "./context/TrainingContext";
import { ThemeProvider } from "./context/ThemeContext";
import { Sidebar } from "./components/Sidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const sidebarWidth = isSidebarExpanded ? '250px' : '68px';

  const hamburgerStyle: React.CSSProperties = {
      position: 'fixed',
      top: '15px',
      left: '15px',
      zIndex: 20, // Higher than the sidebar's z-index
      background: 'var(--card-background)',
      border: '1px solid var(--card-border)',
      color: 'var(--foreground)',
      fontSize: '24px',
      cursor: 'pointer',
      padding: '5px 8px',
      borderRadius: '6px',
      lineHeight: 1,
  };

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <title>swapnavue - The Living Dynamo</title>
        <meta name="description" content="An experimental AI aiming for continuous self-cultivation." />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider>
          <TrainingProvider>
            <div>
              {/* The hamburger button now lives here, outside the sidebar */}
              <button onClick={() => setIsSidebarExpanded(!isSidebarExpanded)} style={hamburgerStyle} aria-label="Toggle navigation">
                ☰
              </button>

              <Sidebar isExpanded={isSidebarExpanded} />
              
              <main style={{ 
                flexGrow: 1, 
                marginLeft: sidebarWidth, 
                padding: '1rem',
                transition: 'margin-left 0.3s ease-in-out',
              }}>
                {children}
              </main>
            </div>
          </TrainingProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Atlas PDF Chat",
  description: "Grounded PDF Q&A workspace for the datco assignment"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}

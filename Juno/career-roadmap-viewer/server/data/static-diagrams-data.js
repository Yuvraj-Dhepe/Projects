/**
 * Static diagrams data
 * This file contains static diagram data for use when file system access fails
 */

export const staticDiagramsMmd = {
  software_engineer: `
graph TD
    A[High School] --> B[Bachelor's in Computer Science]
    B --> C[Junior Software Engineer]
    C --> D[Software Engineer]
    D --> E[Senior Software Engineer]
    E --> F[Lead Software Engineer]
    F --> G[Software Architect]
    G --> H[Engineering Manager]
    H --> I[Chief Technology Officer]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
  `,
  
  data_scientist: `
graph TD
    A[High School] --> B[Bachelor's in Statistics/CS]
    B --> C[Master's in Data Science]
    C --> D[Junior Data Scientist]
    D --> E[Data Scientist]
    E --> F[Senior Data Scientist]
    F --> G[Lead Data Scientist]
    G --> H[Data Science Manager]
    H --> I[Chief Data Officer]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
  `,
  
  cybersecurity_analyst: `
graph TD
    A[High School] --> B[Bachelor's in Cybersecurity]
    B --> C[Security Analyst]
    C --> D[Cybersecurity Specialist]
    D --> E[Senior Security Analyst]
    E --> F[Security Architect]
    F --> G[Chief Information Security Officer]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
  `,
  
  cloud_architect: `
graph TD
    A[High School] --> B[Bachelor's in Computer Science]
    B --> C[Cloud Engineer]
    C --> D[Cloud Architect]
    D --> E[Senior Cloud Architect]
    E --> F[Cloud Solutions Architect]
    F --> G[Chief Cloud Architect]
    G --> H[Chief Technology Officer]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
  `,
  
  devops_engineer: `
graph TD
    A[High School] --> B[Bachelor's in Computer Science]
    B --> C[Junior DevOps Engineer]
    C --> D[DevOps Engineer]
    D --> E[Senior DevOps Engineer]
    E --> F[DevOps Architect]
    F --> G[DevOps Manager]
    G --> H[Chief DevOps Officer]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
  `
};

// Generic diagram for any career path not specifically defined
export function getGenericDiagramMmd(careerPath) {
  // Format the career path name
  const formattedName = careerPath
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
  
  return `
graph TD
    A[High School] --> B[Bachelor's Degree]
    B --> C[Entry Level ${formattedName}]
    C --> D[Mid-Level ${formattedName}]
    D --> E[Senior ${formattedName}]
    E --> F[Lead ${formattedName}]
    F --> G[Management]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
  `;
}

// Function to get a diagram for a specific career path
export function getDiagramMmd(careerPath) {
  if (staticDiagramsMmd[careerPath]) {
    return staticDiagramsMmd[careerPath];
  }
  
  // Return a generic diagram if the specific one doesn't exist
  return getGenericDiagramMmd(careerPath);
}

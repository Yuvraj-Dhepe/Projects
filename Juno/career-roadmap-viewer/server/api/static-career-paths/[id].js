/**
 * Static career paths API endpoint
 * This is a fallback for when file system access fails
 */

// Check if we're in debug mode
const config = useRuntimeConfig();
const DEBUG = config.debugMode === 'true';

/**
 * Debug logger that only logs when DEBUG is true
 * @param {string} message - The message to log
 * @param {...any} args - Additional arguments to log
 */
function debugLog(message, ...args) {
  if (DEBUG) {
    console.log(`[STATIC CAREER PATHS API] ${message}`, ...args);
  }
}

// Static career paths data
const staticCareerPaths = {
  software_engineer: {
    id: 'software_engineer',
    name: 'Software Engineer',
    content: `
# Software Engineer Career Path

## Overview
Software engineering is a field that focuses on designing, developing, and maintaining software systems. Software engineers apply engineering principles to create software solutions that solve real-world problems.

## Requirements
- Bachelor's degree in Computer Science, Software Engineering, or related field
- Strong programming skills in languages like Java, Python, C++, JavaScript
- Understanding of data structures and algorithms
- Knowledge of software development methodologies
- Problem-solving and analytical thinking skills

## Career Progression
1. Junior Software Engineer
2. Software Engineer
3. Senior Software Engineer
4. Lead Software Engineer
5. Software Architect
6. Engineering Manager
7. Chief Technology Officer (CTO)

## Organizations & Certifications
- IEEE Computer Society
- Association for Computing Machinery (ACM)
- AWS Certified Developer
- Microsoft Certified: Azure Developer Associate
- Oracle Certified Professional, Java SE Programmer

## Tips & Resources
- Build a strong portfolio of projects
- Contribute to open-source projects
- Stay updated with the latest technologies and frameworks
- Develop soft skills like communication and teamwork
- Join developer communities and attend tech conferences
`
  },
  data_scientist: {
    id: 'data_scientist',
    name: 'Data Scientist',
    content: `
# Data Scientist Career Path

## Overview
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.

## Requirements
- Bachelor's or Master's degree in Data Science, Statistics, Computer Science, or related field
- Strong programming skills in Python, R, or SQL
- Knowledge of statistical analysis and machine learning algorithms
- Experience with data visualization tools
- Understanding of big data technologies

## Career Progression
1. Junior Data Scientist
2. Data Scientist
3. Senior Data Scientist
4. Lead Data Scientist
5. Data Science Manager
6. Chief Data Officer (CDO)

## Organizations & Certifications
- Data Science Association
- International Association for Statistical Computing
- IBM Data Science Professional Certificate
- Microsoft Certified: Azure Data Scientist Associate
- Google Professional Data Engineer

## Tips & Resources
- Work on real-world data science projects
- Participate in data science competitions (Kaggle)
- Stay updated with the latest algorithms and techniques
- Develop domain expertise in a specific industry
- Network with other data professionals
`
  },
  cybersecurity_analyst: {
    id: 'cybersecurity_analyst',
    name: 'Cybersecurity Analyst',
    content: `
# Cybersecurity Analyst Career Path

## Overview
Cybersecurity analysts protect computer systems and networks from information disclosure, theft, or damage to hardware, software, or electronic data.

## Requirements
- Bachelor's degree in Cybersecurity, Computer Science, or related field
- Knowledge of security frameworks and compliance regulations
- Understanding of network security and threat detection
- Familiarity with security tools and technologies
- Problem-solving and analytical thinking skills

## Career Progression
1. Security Analyst
2. Cybersecurity Specialist
3. Senior Security Analyst
4. Security Architect
5. Chief Information Security Officer (CISO)

## Organizations & Certifications
- International Information System Security Certification Consortium (ISC)²
- SANS Institute
- Certified Information Systems Security Professional (CISSP)
- Certified Ethical Hacker (CEH)
- CompTIA Security+

## Tips & Resources
- Stay updated with the latest security threats and vulnerabilities
- Practice ethical hacking in controlled environments
- Develop a strong understanding of both offensive and defensive security
- Build a network of security professionals
- Attend security conferences and workshops
`
  },
  cloud_architect: {
    id: 'cloud_architect',
    name: 'Cloud Architect',
    content: `
# Cloud Architect Career Path

## Overview
Cloud architects design and oversee an organization's cloud computing strategy, including cloud adoption plans, cloud application design, and cloud management and monitoring.

## Requirements
- Bachelor's degree in Computer Science, Information Technology, or related field
- Strong understanding of cloud platforms (AWS, Azure, Google Cloud)
- Knowledge of networking, security, and infrastructure
- Experience with cloud deployment and migration
- Problem-solving and communication skills

## Career Progression
1. Cloud Engineer
2. Cloud Architect
3. Senior Cloud Architect
4. Cloud Solutions Architect
5. Chief Cloud Architect
6. Chief Technology Officer (CTO)

## Organizations & Certifications
- Cloud Security Alliance
- AWS Certified Solutions Architect
- Microsoft Certified: Azure Solutions Architect
- Google Cloud Certified - Professional Cloud Architect
- IBM Certified Solution Architect - Cloud Computing

## Tips & Resources
- Gain hands-on experience with multiple cloud platforms
- Stay updated with the latest cloud technologies and services
- Develop a strong understanding of security and compliance in the cloud
- Build expertise in cloud migration strategies
- Network with other cloud professionals
`
  },
  devops_engineer: {
    id: 'devops_engineer',
    name: 'DevOps Engineer',
    content: `
# DevOps Engineer Career Path

## Overview
DevOps engineers combine software development and IT operations to shorten the systems development life cycle while delivering features, fixes, and updates frequently in close alignment with business objectives.

## Requirements
- Bachelor's degree in Computer Science, Information Technology, or related field
- Knowledge of programming languages and scripting
- Experience with CI/CD pipelines and automation tools
- Understanding of infrastructure as code
- Familiarity with containerization and orchestration

## Career Progression
1. Junior DevOps Engineer
2. DevOps Engineer
3. Senior DevOps Engineer
4. DevOps Architect
5. DevOps Manager
6. Chief DevOps Officer

## Organizations & Certifications
- DevOps Institute
- AWS Certified DevOps Engineer
- Microsoft Certified: DevOps Engineer Expert
- Docker Certified Associate
- Kubernetes Certified Administrator

## Tips & Resources
- Gain experience with both development and operations
- Learn automation and infrastructure as code
- Develop a strong understanding of CI/CD pipelines
- Stay updated with the latest DevOps tools and practices
- Build a portfolio of DevOps projects
`
  }
};

/**
 * API endpoint to serve static career path information
 * This is a fallback for when file system access fails
 *
 * @param {Object} event - The event object
 * @return {Object} The career path information
 */
export default defineEventHandler(async (event) => {
  try {
    // Get the career path ID from the URL parameter
    const id = event.context.params.id;
    debugLog(`Requested static career path info for: ${id}`);
    
    if (!id) {
      debugLog('No ID provided');
      return {
        statusCode: 400,
        body: 'Career path ID is required'
      };
    }
    
    // Check if we have static data for this career path
    if (staticCareerPaths[id]) {
      debugLog(`Found static data for career path: ${id}`);
      
      // Get the format from query parameter (default to json)
      const query = getQuery(event);
      const format = query.format || 'json';
      
      if (format === 'html') {
        // Return the content as HTML
        debugLog('Returning HTML content');
        return {
          html: staticCareerPaths[id].content
        };
      } else {
        // Return the data as JSON
        debugLog('Returning JSON data');
        return staticCareerPaths[id];
      }
    }
    
    // If we don't have static data for this career path, generate a placeholder
    debugLog(`No static data found for career path: ${id}, generating placeholder`);
    
    // Generate a formatted name from the ID
    const formattedName = id
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
    
    // Generate placeholder content
    const placeholderContent = `
# ${formattedName} Career Path

## Overview
This is a placeholder for the ${formattedName} career path information.

## Requirements
- Education requirements will be listed here
- Skill requirements will be listed here
- Experience requirements will be listed here

## Career Progression
1. Entry-level positions
2. Mid-level positions
3. Senior-level positions
4. Leadership positions

## Organizations & Certifications
- Relevant organizations will be listed here
- Relevant certifications will be listed here

## Tips & Resources
- Career tips will be listed here
- Useful resources will be listed here
`;
    
    // Get the format from query parameter (default to json)
    const query = getQuery(event);
    const format = query.format || 'json';
    
    if (format === 'html') {
      // Return the content as HTML
      debugLog('Returning HTML placeholder content');
      return {
        html: placeholderContent
      };
    } else {
      // Return the data as JSON
      debugLog('Returning JSON placeholder data');
      return {
        id: id,
        name: formattedName,
        content: placeholderContent
      };
    }
  } catch (error) {
    console.error('Error serving static career path info:', error);
    debugLog('Error serving static career path info:', error.message, error.stack);
    
    // Set status code and return error
    return {
      statusCode: 500,
      body: 'Error serving career path information'
    };
  }
});

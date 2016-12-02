---
layout: post
title: Git Commonly Used Commands, Tricks and Work Flows
excerpt: "A note about daily git usage."
categories: [Git]
tags: [code, git, version control, cooperation]
comments: true
---

# Git and Github

## Working with SSH URLs

The Advantages of SSH URLs are super obvious, the trusted computers will no longer needed a password to pull or push. Following will go through how to establish an SSH key for new computer that will be trusted to access the user's repository.

#### Generate an SSH key

- Checking for existing SSH keys. Public key usually has extension _*.pub_.  

  ```bash  
  $ ls -al ~/.ssh
  # Lists the files in your .ssh directory, if they exist
  ```

- If there is no any public key pair, generating a new SSH key and adding it to ssh-agent.  

  ```bash  
  $ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

  Generating public/private rsa key pair.
  Enter a file in which to save the key (/Users/you/.ssh/id_rsa): [Press enter]
  Enter passphrase (empty for no passphrase): [Type a passphrase]
  Enter same passphrase again: [Type passphrase again]
  ```  
  Passphrase is a second layer encryption, which should be set for the sake of security.

- Adding SSH key to ssh-agent.  

  ```bash  
  # start the ssh-agent in the background
  $ eval "$(ssh-agent -s)"
  Agent pid 59566
  # add ssh key to the ssh-agent
  $ ssh-add ~/.ssh/id_rsa
  ```

- Adding a new SSH public key to GitHub account.

  ```bash  
  $ pbcopy < ~/.ssh/id_rsa.pub
  # Copies the contents of the id_rsa.pub file to your clipboard
  ```

  follow the steps below to add the key to account:

  - Click Setting  

  ![Setting]({{ site.url }}/assets/githubpost/userbar-account-settings.png)

  - Click **SSH and GPG keys**  

  ![Setting]({{ site.url }}/assets/githubpost/settings-sidebar-ssh-keys.png)

  - Click **New SSH key** or **Add SSH key**  

  ![Setting]({{ site.url }}/assets/githubpost/ssh-add-ssh-key.png)

  - Paste public key into "Key"

  ![Setting]({{ site.url }}/assets/githubpost/ssh-key-paste.png)

  - Click **Add SSH Key**

  ![Setting]({{ site.url }}/assets/githubpost/ssh-add-key.png)

  - Confirm
  
  ![Setting]({{ site.url }}/assets/githubpost/sudo_mode_popup.png)

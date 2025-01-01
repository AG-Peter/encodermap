# Hosting EncoderMap

In this tutorial, you will be guided through the necessary steps to host EncoderMap's webpage on your computer. As a requisite, we want to have a public IPv4.

## Determining whether you have a public IPv4

Run this command to get your public IPv4:

```bash
curl https://ipinfo.io/ip
```

Note down the output. Then run this command to get your local IPv4:

```bash
ifconfig | grep inet
```

If this output contains the same IP address, as the `curl` command, we are set. Otherwise, your are sitting behind a router and want to add a port forwarding in your router configuration to your local IPv4 address. Forward both ports 80 and 443 using the TCP protocol.

## Installing Apache

To install apache2 run:

```bash
sudo apt install apache2
```

Visit the IP address from the `curl` command in your browser. If you see the Apache2 default page you are set.

### Extra: Prevent search engines from crawling your site

EncoderMap's Sphinx documentation comes with its own `robots.txt` which prevents all search engines from crawling and indexing the site.

```
User-agent: *
Disallow: /
```

Of course, if you want the page to be visible, you need to remove that file from `docs/source/_static/robots.txt` and/or the setting in `docs/source/conf.py`. The `run_docbuild_test_and_cover.py` script has an option that deletes the `robots.txt` after a pull.

Finally, we will add apache2 to the autostart:

```bash
sudo update-rc.d apache2 defaults
```

## Creating passwordless ssh to GitHub, GitLab

### Creating a directory on the server

If you are running on a shared home, we recommend to create a copy of the repository on a local drive, where constant builds and rebuilds don't affect your data-traffic too much.

```bash
cd /mnt/data
git clone git https://github.com/AG-Peter/encodermap.git
cd encodermap
```

### Setting up ssh keys

For the cron job, it might be beneficial to allow passwordless access to EncoderMap's GitHub and GitLab repository. To do so for GitHub, we first create a new ssh key using (leave the passphrase empty):

```bash
cd /mnt/data
ssh-keygen -t ed25519 -C "your.github.email@provider.com"" -f encodermap_passwordless
```

The public key of this key-pair then needs to be uploaded to GitHub. Get the content with:

```bash
cat encodermap_passwordless.pub
```

### GitHub

And add it on this site: https://github.com/settings/keys (after logging in to GitHub).

Then, we need to tell the `git clone` command to use this key for cloning/pulling. For that we need to configure the ssh settings in `~/.ssh/config` and add:

```
Host encodermap_passwordless_github
     HostName github.com
     User git
     IdentityFile /mnt/data/encodermap_passwordless
```
We can test the connection via:

```bash
ssh -T git@encodermap_passwordless_github
```
Then, we can clone via:

```angular2html
git clone git@encodermap_passwordless_github:AG-Peter/encodermap.git
```

To better distinguish GitHub from GitLab, we rename the remote:

```bash
git remote rename origin public
```

### GitLab

To do the same thing with GitLab, we add this configuration to `~/.ssh/conf`:

```
Host encodermap_passwordless_gitlab
     HostName gitlab.inf.uni-konstanz.de
     User git
     IdentityFile /mnt/data/encodermap_passwordless
     IdentitiesOnly yes
```

And upload the public key to GitLab. We then add the new remote via:

```bash
git remote add gitlab git@encodermap_passwordless_gitlab:ag-peter/encoder_map_private.git
```

We can now switch between the public encodermap (changes in branch `main`) and the private repo on GitLab.

## Building the documentation

We can now install the requirements

```bash
pip install -r requirements.txt
pip install -r md_equirements.txt
pip install -r docs/sphinx_equirements.txt
pip install -r tests/test_equirements.txt
```

We then build the documentation with the `run_docbuild_test_and_cover.py` script. For our purposes, we can skip the tests so that the script concludes in shorter time (Executing the notebooks can take a while, too. But they are only executed when they change between commits).

```bash
./docs/run_docbuild_test_and_cover.py --doc-only
```

This will create a directory under `docs/build` named `html`. This is where the page resides. We then need to tell apache2 to host the site from this directory.

## Linking EncoderMap's docs and apache2

Inside the apache directory (`/var/www/html`) we then create a softlink to EncoderMap's build documentation.

```bash
sudo ln -s /mnt/data/git/encodermap/docs/build/html html_public
```

We then need to set the document root for the site in apache, by a line in apache2's default site: `/etc/apache2/sites-available/000-default.conf`

Change DocumentRoot from `DocumentRoot /var/www/html` to `DocumentRoot /var/www/html/html_public` and activate the new site by either reloading apache2:

```bash
sudo /etc/init.d/apache2 reload
```

or restarting

```bash
sudo service apache2 restart
```

Then open your public IP in a browser, and you should be greeted by EncoderMap's documentation.

## Generating a certificate

We want to change the site from http to https. For that, we call first:

```bash
sudo a2enmod ssl
```

Then, we create the certificate:

```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/apache-selfsigned.key -out /etc/ssl/certs/apache-selfsigned.crt
```

and answer the questions as good, as we can. Providing our public IP for the server's common name. When the server is accessible via a FQDN (fully qualified domain name), we can use certbot to request certificates signed by a certificate authority and skip the safety warning.

We then change `/etc/apache2/sites-available/000-default.conf` again. We change the port of the first virtual host from `*:80` to `*:443` and add the ssl settings. We then add another virtualhost, that redirects http to https.

```
<VirtualHost *:443>

    # Here's the other stuff we already configured.

   SSLEngine on
   SSLCertificateFile /etc/ssl/certs/apache-selfsigned.crt
   SSLCertificateKeyFile /etc/ssl/private/apache-selfsigned.key
</VirtualHost>
```

And create a new site called `001-redirect.conf`

```
<VirtualHost *:80>
	ServerName your_domain_or_ip
	Redirect / https://your_domain_or_ip/
</VirtualHost>
```

to the configuration of the virtual host. We can then test and (after a successful test) reload apache.

```bash
sudp a2ensite 001-redirect.conf
sudo apachectl configtest
sudo systemctl reload apache2
```

## Using a FQDN

For the purposes of this tutorial, we have bough the domain `encodermap.site` and forward it via an A record to our public IP. We then need to reconfigure Apache to use this site and then can use certbot to get a trusted certificate.

Install certbot via

```bash
sudo apt install certbot python3-certbot-apache
```

We then check the configuration of the virtual host in `/etc/apache2/sites-available/000-default.conf` and put our new domain into the ServerName fields.

We then get our certificate via:

```bash
sudo certbot --apache
```

## Linking to EncoderMap's data repos

### Creating .htaccess

### Setting passwords

## Running a cronjob
